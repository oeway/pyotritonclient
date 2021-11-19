import asyncio
import imageio
from pyotritonclient import execute


async def run():
    image = imageio.imread(
        "https://raw.githubusercontent.com/stardist/stardist/3451a4f9e7b6dcef91b09635cc8fa78939fb0d29/stardist/data/images/img2d.tif"
    )
    image = image.astype("uint16")
    param = {}

    # run inference
    results = await execute(inputs=[image, param], server_url='https://ai.imjoy.io/triton', model_name='stardist', decode_bytes=True)
    mask = results["mask"]
    assert mask.shape == (512, 512)


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
