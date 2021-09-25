import asyncio
import io
import imageio
import numpy as np
from pyotritonclient import get_config, execute_model


async def run():
    # obtain the model config
    config = await get_config("https://triton.imjoy.io", "stardist")

    image = imageio.imread(
        "https://raw.githubusercontent.com/stardist/stardist/3451a4f9e7b6dcef91b09635cc8fa78939fb0d29/stardist/data/images/img2d.tif"
    )
    image = image.astype("uint16")
    param = {}

    # run inference
    results = await execute_model([image, param], config=config, decode_bytes=True)
    mask = results["mask"]
    assert mask.shape == (512, 512)


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
