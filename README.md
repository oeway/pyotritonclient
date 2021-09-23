# Triton HTTP Client for Pyodide

A Pyodide python http client library and utilities for communicating with Triton Inference Server (based on tritonclient from NVIDIA).


This is a simplified implemetation of the triton client from NVIDIA, it is mainly made for running in the web browser with pyodide.
It only implement the http client, and most of the API remains the same but changed into async.

## Usage

You can used it in Pyodide based environment, for example: [Pyodide console](https://pyodide-cdn2.iodide.io/dev/full/console.html) or [JupyterLite](https://jupyterlite.readthedocs.io/en/latest/_static/lab/index.html).

For example in a JupyterLite notebook, you can install the client by running:
```python
import micropip
micropip.install("pyotritonclient")
```

If you want to use outside pyodide, run `pip install pyotritonclient`. However, please note that you will need to provide your own http_client class.

To execute the model, we provide utility functions to make it much easier:
```python
import numpy as np
from pyotritonclient import get_config, execute_model

# obtain the model config
config = await get_config('https://triton.imjoy.io', 'cellpose-cyto')

# create fake input tensors
input0 = np.zeros([2, 349, 467], dtype='float32')
input1 = np.array([30], dtype='float32')
# run inference
results = await execute_model([input0, input1], config=config)
```

You can access the lower level api, see the [test example](./tests/test_client.py).

You can also find the official [client examples](https://github.com/triton-inference-server/client/tree/main/src/python/examples) demonstrate how to use the 
package to issue request to [triton inference server](https://github.com/triton-inference-server/server). However, please notice that you will need to
change the http client code into async style. For example, instead of doing `client.infer(...)`, you need to do `await client.infer(...)`.

The http client code is forked from [triton client git repo](https://github.com/triton-inference-server/client) since commit [b3005f9db154247a4c792633e54f25f35ccadff0](https://github.com/triton-inference-server/client/tree/b3005f9db154247a4c792633e54f25f35ccadff0).


## Server setup
Since we access the server from the browser environment which typically has more security restrictions, it is important that the server is configured to enable browser access.

Please make sure you configured following aspects:
 * The server must provide HTTPS endpoints instead of HTTP
 * The server should send the following headers:
    - `Access-Control-Allow-Headers: Inference-Header-Content-Length,Accept-Encoding,Content-Encoding,Access-Control-Allow-Headers`
    - `Access-Control-Expose-Headers: Inference-Header-Content-Length,Range,Origin,Content-Type`
    - `Access-Control-Allow-Methods: GET,HEAD,OPTIONS,PUT,POST`
    - `Access-Control-Allow-Origin: *` (This is optional depending on whether you want to support CORS)
