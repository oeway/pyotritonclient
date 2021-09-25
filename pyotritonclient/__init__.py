import json
import numpy as np
from pyotritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
import pyotritonclient.http as httpclient


async def get_config(server_url, model_name, verbose=False):
    """
    Function for getting model config
    """
    with httpclient.InferenceServerClient(server_url, verbose=verbose) as client:
        response = await client.get_model_config(model_name)
        response["server_url"] = server_url
        response["model_name"] = model_name
        return response


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
    verbose=False,
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

    for i in range(len(inputs)):
        # automatically encode string and dict into np.bytes_
        if isinstance(inputs[i], str):
            bytes_data = str.encode(inputs[i], "utf-8")
            inputs[i] = np.array([bytes_data], dtype=np.object_)
        if isinstance(inputs[i], dict):
            # encode the dictionary as as np.object_
            bytes_data = str.encode(json.dumps(inputs[i]), "utf-8")
            inputs[i] = np.array([bytes_data], dtype=np.object_)

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
        )

        info = response.get_response()
        results = {
            output["name"]: response.as_numpy(output["name"])
            for output in info["outputs"]
        }

        if decode_bytes:
            for k in results:
                # decode bytes to utf-8
                if results[k].dtype == np.object_:
                    results[k] = results[k].astype(np.bytes_)
                    results[k] = [
                        str(np.char.decode(results[k][i], "UTF-8"))
                        for i in range(len(results[k]))
                    ]

        results["__info__"] = info
        return results
