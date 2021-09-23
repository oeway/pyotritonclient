import numpy as np
from pyotritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
import pyotritonclient.http as httpclient


async def get_config(server_url, model_name):
    """
    Function for getting model config
    """
    with httpclient.InferenceServerClient(server_url, verbose=True) as client:
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
    input_types = [inputc[i]["data_type"].lstrip("TYPE_") for i in range(len(inputc))]

    for i in range(len(inputs)):
        if inputs[i].dtype != triton_to_np_dtype(input_types[i]):
            raise TypeError(
                f"Input ({i}) data type mismatch: {inputs[i].dtype} (should be {triton_to_np_dtype(input_types[i])})"
            )

    if select_outputs:
        for out in select_outputs:
            if out not in output_names:
                raise Exception(f"Output name {out} does not exist.")
    else:
        select_outputs = output_names

    with httpclient.InferenceServerClient(server_url, verbose=True) as client:
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

        results = await client.infer(
            model_name,
            infer_inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            request_compression_algorithm=compression_algorithm,
            response_compression_algorithm=compression_algorithm,
        )

        return {
            output["name"]: results.as_numpy(output["name"])
            for output in results._result["outputs"]
        }
