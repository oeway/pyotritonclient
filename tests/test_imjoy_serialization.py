from imjoy_rpc.hypha import RPC
import msgpack
import numpy as np
import asyncio
from pyotritonclient import execute

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
    results = await execute(
        [kwargs],
        server_url="http://localhost:5000",
        model_name="bioengine-model-runner",
        serialization="imjoy",
    )
    result = results["outputs"]
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed")


asyncio.run(test_model())
