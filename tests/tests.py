"""
Run HPA bestfitting densenet model with a remote triton inference server

You can download the model package from here: 
https://github.com/CellProfiling/densenet/releases/download/v0.1.0/bestfitting-densenet_triton_onnx.zip

To run this demo, please install tritonclient and imageio:
```
pip install tritonclient[http] opencv-python
```


Author: Wei OUYANG (oeway007@gmail.com)
"""
import numpy as np
import cv2
import time
from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient

import os
import urllib.request

COLORS =  ["red", "green", "blue", "yellow"]
LABELS = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
}

def read_rgby(work_dir, image_id, crop_size=1024, suffix='jpg'):
    image = [
        cv2.imread(
            os.path.join(work_dir, "%s_%s.%s" % (image_id, color, suffix)),
            cv2.IMREAD_GRAYSCALE,
        )
        for color in COLORS
    ]
    image = np.stack(image, axis=-1)
    h, w = image.shape[:2]
    if crop_size != h or crop_size != w:
        image = cv2.resize(
            image,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
    return image

# TODO: fix fetch image to use pyodide
def fetch_image(work_dir, image_url):
    img_name = image_url.split('/')[-1]
    fpath = os.path.join(work_dir, img_name)
    if not os.path.exists(fpath):
        urllib.request.urlretrieve(image_url, fpath)
    return fpath

async def run_inference(
    img,
    request_id="",
    server_url="europa.scilifelab.se:8000",
    model_name="bestfitting-inceptionv3-single-cell",
    model_version="",
    compression_algorithm='deflate',
    ):

    with httpclient.InferenceServerClient(server_url) as client:
        response = await client.get_model_config(model_name, model_version=model_version)
        assert len(response['input']) == 1 and len(response['output']) == 2
        input0_name = response['input'][0]['name']
        output0_name = response['output'][0]['name']
        output1_name = response['output'][1]['name']
        
        input0_data = img.astype(np.float32)
        
        inputs = [
            httpclient.InferInput(input0_name, input0_data.shape,
                                np_to_triton_dtype(input0_data.dtype)),
        ]

        inputs[0].set_data_from_numpy(input0_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput(output0_name), #, class_count=28
            httpclient.InferRequestedOutput(output1_name),
        ]

        start = time.time()
        response = await client.infer(model_name,
                                inputs,
                                model_version=model_version,
                                request_id=request_id,
                                outputs=outputs,
                                request_compression_algorithm=compression_algorithm,
                                response_compression_algorithm=compression_algorithm)

        # result = response.get_response()
        output0_data = response.as_numpy(output0_name)
        output1_data = response.as_numpy(output1_name)
        total_time = time.time() - start

        print("{} ({}) => {} ({}) / {} ({}), execution time: {:.2f}s".format(
            input0_name, input0_data.shape, output0_name, output0_data.shape, output1_name, output1_data[0], total_time))
        return output0_data, output1_data

async def predict_single_cells(
    images,
    work_dir = './data',
    threshold = 0.3,
    image_size=128
    ):
    os.makedirs(work_dir, exist_ok=True)
    for image_url in images:
        fpath = fetch_image(work_dir, image_url)
        input_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype('float32')
        input_image = cv2.resize(input_image, (image_size, image_size))
        input_image = input_image.transpose(2, 0, 1)
        input_image = input_image.reshape(4, input_image.shape[1], input_image.shape[2])
        classes, features = await run_inference(input_image / 255.0)
        pred = [(LABELS[i], prob) for i, prob in enumerate(classes.tolist()) if prob>threshold]
        print('Image:', fpath, 'Prediction:', pred, 'Features:', features)

if __name__ == '__main__':
    images = [
        'https://github.com/oeway/hpa-bestfitting-inceptionv3/releases/download/v0.1.0/115_672_E2_1_6.png',
        'https://github.com/oeway/hpa-bestfitting-inceptionv3/releases/download/v0.1.0/115_672_E2_1_10.png']
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(predict_single_cells(images))
    # The result should be similar to the following (obtained with native onnxruntime):
    # 115_672_E2_1 6 [('Cytosol', 0.5525864958763123)] (1, 2048)
    # 115_672_E2_1 10 [('Vesicles and punctate cytosolic patterns', 0.3182237446308136), ('Negative', 0.4854247570037842)] (1, 2048)

