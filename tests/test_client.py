import pickle
import urllib.request

import imageio
import numpy as np
import pytest
from pyotritonclient import SequenceExcutor, execute, execute_batch

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


@pytest.fixture(name="example_dataset", scope="session")
def example_dataset():
    # Download the file from `url` and save it locally under `file_name`:
    urllib.request.urlretrieve(
        "https://github.com/imjoy-team/imjoy-interactive-segmentation/releases/download/v0.1.0/train_samples_4.pkl",
        "train_samples_4.pkl",
    )
    train_samples = pickle.load(open("train_samples_4.pkl", "rb"))
    print(len(train_samples))
    yield train_samples[:2]


async def test_execute_batch():
    image = np.random.randint(0, 255, size=(1, 256, 256), dtype=np.uint8).astype(
        "float32"
    )

    def print_result(a, b):
        print(a)

    results = await execute_batch(
        [[image, {}]] * 10,
        batch_size=2,
        on_batch_end=print_result,
        server_url="https://ai.imjoy.io/triton",
        model_name="bioimageio-dsb-nuclei-boundary",
    )
    print(len(results))
    return results


async def test_sequence_batch(example_dataset):
    seq = SequenceExcutor(
        server_url="https://ai.imjoy.io/triton",
        model_name="cellpose-train",
        decode_json=True,
        sequence_id=10,
    )
    input_sequence = []
    for (image, labels, info) in example_dataset:
        inputs = [
            image.astype("float32"),
            labels.astype("uint16"),
            {"steps": 1, "resume": True},
        ]
        input_sequence.append(inputs)
    results = await seq(input_sequence)
    assert len(results) == len(input_sequence)


async def test_sequence(example_dataset):
    epochs = 1
    seq = SequenceExcutor(
        server_url="https://ai.imjoy.io/triton",
        model_name="cellpose-train",
        decode_json=True,
        sequence_id=10,
    )
    for epoch in range(epochs):
        losses = []
        for (image, labels, info) in example_dataset:
            inputs = [
                image.astype("float32"),
                labels.astype("uint16"),
                {"steps": 1, "resume": True},
            ]
            result = await seq.step(inputs)
            losses.append(result["info"][0]["loss"])
        avg_loss = np.array(losses).mean()
        print(f"Epoch {epoch}  loss={avg_loss}")

    result = await seq.end()
    weights = result["model"][0]
    assert len(weights) > 1000


async def test_execute():
    image = imageio.imread(
        "https://raw.githubusercontent.com/stardist/stardist/3451a4f9e7b6dcef91b09635cc8fa78939fb0d29/stardist/data/images/img2d.tif"
    )
    image = image.astype("uint16")
    param = {}

    # run inference
    results = await execute(
        inputs=[image, param],
        server_url="https://ai.imjoy.io/triton",
        model_name="stardist",
        decode_bytes=True,
    )
    mask = results["mask"]
    assert mask.shape == (512, 512)
