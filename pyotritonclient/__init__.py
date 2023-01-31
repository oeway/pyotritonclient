import json
import asyncio
from pathlib import Path

import numpy as np

import pyotritonclient.http as httpclient
from pyotritonclient.utils import triton_to_np_dtype


async def get_config(server_url, model_name, verbose=False):
    """
    Function for getting model config
    """
    with httpclient.InferenceServerClient(server_url, verbose=verbose) as client:
        response = await client.get_model_config(model_name)
        response["server_url"] = server_url
        response["model_name"] = model_name
        return response


def _encode_input(data):
    # automatically encode string and dict into np.bytes_
    if isinstance(data, str):
        bytes_data = str.encode(data, "utf-8")
        in_data = np.array([bytes_data], dtype=np.object_)
    elif isinstance(data, dict):
        # encode the dictionary as as np.object_
        bytes_data = str.encode(json.dumps(data), "utf-8")
        in_data = np.array([bytes_data], dtype=np.object_)
    elif isinstance(data, (tuple, list)):
        in_data = np.stack(list(map(_encode_input, data)), axis=0)
    else:
        in_data = data
    return in_data


async def execute_model(
    inputs,
    server_url=None,
    model_name=None,
    config=None,
    select_outputs=None,
    request_id="",
    model_version="",
    compression_algorithm="gzip",
    serialization="triton",
    decode_bytes=False,
    decode_json=False,
    verbose=False,
    **kwargs,
):
    """
    Function for execute the model by passing a list of input tensors
    """
    if config is None:
        if server_url is None or model_name is None:
            raise Exception("Please provide config or server_url + model_name")

        config = await get_config(server_url, model_name)
    else:
        server_url = config["server_url"]
        model_name = config["model_name"]

    inputc, outputc = config["input"], config["output"]
    # Disable length check since the inputs can be optional
    # assert len(inputc) == len(
    #     inputs
    # ), f"Invalid inputs number: {len(inputs)}, it should be {len(inputc)}"
    input_names = [inputc[i]["name"] for i in range(len(inputc))]
    output_names = [outputc[i]["name"] for i in range(len(outputc))]
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    if isinstance(inputs, dict):
        inputs = [inputs[name] for name in input_names]
    assert isinstance(inputs, list), "Inputs must be a list or tuple"

    assert serialization in ["triton", "imjoy"]

    if serialization == "triton":
        input_types = [
            inputc[i]["data_type"].lstrip("TYPE_").replace("STRING", "BYTES")
            for i in range(len(inputc))
        ]

        for i in range(len(inputs)):
            inputs[i] = _encode_input(inputs[i])

            if inputs[i].dtype != triton_to_np_dtype(input_types[i]):
                if not (
                    inputs[i].dtype == np.bytes_
                    and triton_to_np_dtype(input_types[i]) == np.object_
                ):
                    raise TypeError(
                        f"Input ({i}: {input_names[i]}) data type mismatch: {inputs[i].dtype} (should be {triton_to_np_dtype(input_types[i])})"
                    )
    else:
        from imjoy_rpc.hypha import RPC
        import msgpack

        _rpc = RPC(None, "anon")
        input_types = []
        for i in range(len(inputs)):
            data = _rpc.encode(inputs[i])
            bytes_data = msgpack.dumps(data)
            inputs[i] = np.array([bytes_data], dtype=np.object_)
            input_types.append("BYTES")

    if select_outputs:
        for out in select_outputs:
            if out not in output_names:
                raise Exception(f"Output name {out} does not exist.")
    else:
        select_outputs = output_names

    with httpclient.InferenceServerClient(server_url, verbose=verbose) as client:
        infer_inputs = [
            httpclient.InferInput(input_names[i], inputs[i].shape, input_types[i])
            for i in range(len(inputs))
        ]

        for i in range(len(inputs)):
            infer_inputs[i].set_data_from_numpy(inputs[i], binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput(output_names[i])
            for i in range(len(output_names))
            if output_names[i] in select_outputs
        ]

        response = await client.infer(
            model_name,
            infer_inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            request_compression_algorithm=compression_algorithm,
            response_compression_algorithm=compression_algorithm,
            **kwargs,
        )

        info = response.get_response()
        results = {
            output["name"]: response.as_numpy(output["name"])
            for output in info["outputs"]
        }
        if serialization == "triton":
            if decode_bytes or decode_json:
                for k in results:
                    # decode bytes to utf-8
                    if results[k].dtype == np.object_:
                        bytes_array = results[k].astype(np.bytes_)
                        data = []
                        for i in range(len(bytes_array)):
                            try:
                                v = str(np.char.decode(bytes_array[i], "UTF-8"))
                                if decode_json:
                                    try:
                                        v = json.loads(v)
                                    except json.JSONDecodeError:
                                        pass
                            except UnicodeDecodeError:
                                v = results[k][i]
                            data.append(v)
                        results[k] = data
        else:
            assert (
                not decode_bytes and not decode_json
            ), "decode_json or decode_json cannot be True for imjoy serialization method"
            for k in results:
                if results[k].dtype == np.object_:
                    bytes_array = results[k].astype(np.bytes_)[0].item()
                    data = msgpack.loads(bytes_array)
                    results[k] = _rpc.decode(data)

        results["__info__"] = info
        return results


_config_cache = {}


async def execute(
    inputs=None, server_url=None, model_name=None, cache_config=True, **kwargs
):
    """
    Function for execute the model by passing a list of input tensors and using cached config
    """
    if cache_config:
        if (server_url, model_name) not in _config_cache:
            config = await get_config(server_url, model_name)
            _config_cache[(server_url, model_name)] = config
        else:
            config = _config_cache[(server_url, model_name)]
    else:
        config = await get_config(server_url, model_name)
        _config_cache[(server_url, model_name)] = config
    return await execute_model(inputs, config=config, **kwargs)


async def execute_batch(batches, batch_size=1, on_batch_end=None, **kwargs):
    """
    Function for execute the model by batching inputs
    """
    results = []
    for b in range(0, len(batches), batch_size):
        futs = []
        for i in range(batch_size):
            if b + i >= len(batches):
                break
            futs.append(execute(inputs=batches[b + i], **kwargs))
        if len(futs) > 0:
            batch_results = await asyncio.gather(*futs)
            results.extend(batch_results)
            if on_batch_end:
                on_batch_end(b, batch_results)
    return results


class SequenceExcutor:
    """Execute a sequence by managing the sequence_start and sequence_end automatically"""

    def __init__(self, start=True, **kwargs) -> None:
        self.kwargs = kwargs
        if "sequence_id" not in kwargs:
            raise Exception("Please provide sequence_id")
        self._seq_start = start
        self._last_args = None

    def reset(self):
        self._seq_start = True
        self._last_args = None

    async def __call__(self, inputs_batch, on_step=None, **kwargs):
        assert isinstance(inputs_batch, (tuple, list))
        if len(inputs_batch) == 1:
            assert isinstance(inputs_batch[0], (tuple, list))
            result = await execute(
                inputs_batch[0], sequence_start=True, sequence_end=True, **kwargs
            )
            self._seq_start = True
        else:
            results_batch = []
            for i, inputs in enumerate(inputs_batch[:-1]):
                assert isinstance(inputs, (tuple, list))
                result = await self.step(inputs, **kwargs)
                results_batch.append(result)
                if on_step:
                    on_step(i, result)
            result = await self.end(inputs_batch[-1], **kwargs)
        results_batch.append(result)
        return results_batch

    async def step(self, *args, **kwargs):
        if "sequence_start" in kwargs:
            raise Exception(
                "sequence_start are not allowed keywords in sequence executor"
            )
        current_kwargs = {}
        current_kwargs.update(self.kwargs)
        current_kwargs.update(kwargs)
        self._last_args = (args, current_kwargs)
        if self._seq_start:
            self._seq_start = False
            return await execute(*args, sequence_start=True, **current_kwargs)
        else:
            return await execute(*args, sequence_start=False, **current_kwargs)

    async def end(self, *args, **kwargs):
        if "sequence_end" in kwargs:
            raise Exception(
                "sequence_end are not allowed keywords in sequence executor"
            )
        current_kwargs = {}
        current_kwargs.update(self.kwargs)
        if self._last_args:
            (_args, _kwargs) = self._last_args
            if not args:
                args = _args
            current_kwargs.update(_kwargs)
        current_kwargs.update(kwargs)
        if self._seq_start:
            current_kwargs["sequence_start"] = True
        current_kwargs["sequence_end"] = True
        # reset the sequence
        self._seq_start = True
        return await execute(*args, **current_kwargs)


# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]
