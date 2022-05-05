from imjoy_rpc.hypha import RPC
import msgpack
import numpy as np
import asyncio
from pyotritonclient import execute
import imageio

_rpc = RPC(None, "anon")

original_inputs = [{"args": 123, "array": np.zeros([10, 20])}]

inputs = original_inputs.copy()
input_types = []
for i in range(len(inputs)):
    data = _rpc.encode(inputs[i])
    bytes_data = msgpack.dumps(data)
    inputs[i] = np.array(bytes_data, dtype=np.object_)
    input_types.append("BYTES")

results = inputs
for k in range(len(results)):
    if results[k].dtype == np.object_:
        bytes_array = results[k].astype(np.bytes_).item()
        data = msgpack.loads(bytes_array)
        results[k] = _rpc.decode(data)


assert results[0]["args"] == original_inputs[0]["args"]
assert np.all(results[0]["array"] == original_inputs[0]["array"])


async def test_model():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8).astype(
        "float32"
    )
    kwargs = {"inputs": [image], "model_id": "10.5281/zenodo.5869899"}
    ret = await execute(
        [kwargs],
        server_url="http://localhost:8000",
        model_name="bioengine-model-runner",
        serialization="imjoy",
    )
    result = ret["result"]
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed")


async def test_execute():
    image = imageio.imread(
        "https://raw.githubusercontent.com/stardist/stardist/3451a4f9e7b6dcef91b09635cc8fa78939fb0d29/stardist/data/images/img2d.tif"
    )
    image = image.astype("uint16")
    param = {}

    # run inference
    results = await execute(
        inputs=[image, param],
        server_url="http://localhost:8000",
        model_name="stardist",
        decode_bytes=True,
    )
    mask = results["mask"]
    assert mask.shape == (512, 512)
    print("stardist test passed")


asyncio.run(test_execute())
asyncio.run(test_model())
