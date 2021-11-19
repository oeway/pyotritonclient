import json
from pathlib import Path

import numpy as np

import pyotritonclient.http as httpclient
from pyotritonclient.utils import np_to_triton_dtype, triton_to_np_dtype


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
    compression_algorithm="deflate",
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
    assert len(inputc) == len(
        inputs
    ), f"Invalid inputs number: {len(inputs)}, it should be {len(inputc)}"
    input_names = [inputc[i]["name"] for i in range(len(inputc))]
    output_names = [outputc[i]["name"] for i in range(len(outputc))]
    input_types = [
        inputc[i]["data_type"].lstrip("TYPE_").replace("STRING", "BYTES")
        for i in range(len(inputc))
    ]
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    assert isinstance(inputs, list), "Inputs must be a list or tuple"
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

        if decode_bytes or decode_json:
            for k in results:
                # decode bytes to utf-8
                if results[k].dtype == np.object_:
                    results[k] = results[k].astype(np.bytes_)
                    data = []
                    for i in range(len(results[k])):
                        v = str(np.char.decode(results[k][i], "UTF-8"))
                        if decode_json:
                            try:
                                v = json.loads(v)
                            except json.JSONDecodeError:
                                pass
                        data.append(v)
                    results[k] = data

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
    return await execute_model(inputs, config=config, **kwargs)


class SequenceExcutor:
    """Execute a sequence by managing the sequence_start and sequence_end automatically"""

    def __init__(self, auto_end=False, **kwargs) -> None:
        self.kwargs = kwargs
        if "sequence_id" not in kwargs:
            raise Exception("Please provide sequence_id")
        self._seq_start = True
        self._last_args = None
        self._auto_end = auto_end

    async def execute(self, *args, **kwargs):
        if "sequence_start" in kwargs:
            raise Exception(
                "sequence_start are not allowed keywords in sequence executor"
            )
        kwargs.update(self.kwargs)
        if self._auto_end:
            self._last_args = (args, kwargs)
        if self._seq_start:
            self._seq_start = False
            return await execute(*args, sequence_start=True, **kwargs)
        else:
            return await execute(*args, sequence_start=False, **kwargs)

    async def __aenter__(self):
        self._seq_start = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._auto_end and self._last_args:
            (args, kwargs) = self._last_args
            await execute(*args, sequence_start=False, sequence_end=True, **kwargs)


# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]
