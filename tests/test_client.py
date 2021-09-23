import time
import asyncio
import numpy as np
from pyotritonclient.utils import np_to_triton_dtype
import pyotritonclient.http as httpclient


async def run_inference(
    img,
    server_url="https://triton.imjoy.io",
    model_name="cellpose-cyto",
    request_id="",
    model_version="",
    compression_algorithm="deflate",
    diameter=30,
):

    with httpclient.InferenceServerClient(server_url, ssl=True, verbose=True) as client:
        response = await client.get_model_config(model_name)
        # await printobj(config)
        assert len(response["input"]) == 2 and len(response["output"]) == 2
        input0_name = response["input"][0]["name"]
        input1_name = response["input"][1]["name"]
        output0_name = response["output"][0]["name"]
        output1_name = response["output"][1]["name"]

        input0_data = img.astype(np.float32)
        input1_data = np.array([diameter], dtype=np.float32)

        inputs = [
            httpclient.InferInput(
                input0_name, input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            httpclient.InferInput(
                input1_name, input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput(output0_name),
            httpclient.InferRequestedOutput(output1_name),
        ]

        start = time.time()
        response = await client.infer(
            model_name,
            inputs,
            model_version=model_version,
            request_id=request_id,
            outputs=outputs,
            request_compression_algorithm=compression_algorithm,
            response_compression_algorithm=compression_algorithm,
        )

        # result = response.get_response()
        output0_data = response.as_numpy(output0_name)
        output1_data = response.as_numpy(output1_name)
        total_time = time.time() - start

        print(
            "{} ({}) / {} ({}) => {} ({}) / {} ({}), execution time: {:.2f}s".format(
                input0_name,
                input0_data.shape,
                input1_name,
                input1_data,
                output0_name,
                output0_data.shape,
                output1_name,
                output1_data[0],
                total_time,
            )
        )
    return output0_data, output1_data


if __name__ == "__main__":
    image = np.zeros([2, 349, 467])
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_inference(image))
