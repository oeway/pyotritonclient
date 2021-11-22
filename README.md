# Triton HTTP Client for Pyodide

A Pyodide python http client library and utilities for communicating with Triton Inference Server (based on tritonclient from NVIDIA).


This is a simplified implemetation of the triton client from NVIDIA, it works both in the browser with Pyodide Python or the native Python.
It only implement the http client, and most of the API remains the similar but changed into async and with additional utility functions.

## Installation

To use it in native CPython, you can install the package by running:
```
pip install pyotritonclient
```

For Pyodide-based Python environment, for example: [JupyterLite](https://jupyterlite.readthedocs.io/en/latest/_static/lab/index.html) or [Pyodide console](https://pyodide-cdn2.iodide.io/dev/full/console.html), you can install the client by running the following python code:
```python
import micropip
micropip.install("pyotritonclient")
```
## Usage

### Basic example
To execute the model, we provide utility functions to make it much easier:
```python
import numpy as np
from pyotritonclient import execute

# create fake input tensors
input0 = np.zeros([2, 349, 467], dtype='float32')
# run inference
results = await execute(inputs=[input0, {"diameter": 30}], server_url='https://ai.imjoy.io/triton', model_name='cellpose-python')
```

The above example assumes you are running the code in a jupyter notebook or an environment supports top-level await, if you are trying the example code in a normal python script, please wrap the code into an async function and run with asyncio as follows:
```python
import asyncio
import numpy as np
from pyotritonclient import execute

async def run():
    results = await execute(inputs=[np.zeros([2, 349, 467], dtype='float32'), {"diameter": 30}], server_url='https://ai.imjoy.io/triton', model_name='cellpose-python')
    print(results)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
```

You can access the lower level api, see the [test example](./tests/test_client.py).

You can also find the official [client examples](https://github.com/triton-inference-server/client/tree/main/src/python/examples) demonstrate how to use the 
package to issue request to [triton inference server](https://github.com/triton-inference-server/server). However, please notice that you will need to
change the http client code into async style. For example, instead of doing `client.infer(...)`, you need to do `await client.infer(...)`.

The http client code is forked from [triton client git repo](https://github.com/triton-inference-server/client) since commit [b3005f9db154247a4c792633e54f25f35ccadff0](https://github.com/triton-inference-server/client/tree/b3005f9db154247a4c792633e54f25f35ccadff0).


### Using the sequence executor with stateful models
To simplify the manipulation on stateful models with sequence, we also provide the `SequenceExecutor` to make it easier to run models in a sequence.
```python
from pyotritonclient import SequenceExcutor


seq = SequenceExcutor(
  server_url='https://ai.imjoy.io/triton',
  model_name='cellpose-train',
  sequence_id=100
)
inputs = [
  image.astype('float32'),
  labels.astype('float32'),
  {"steps": 1, "resume": True}
]
for (image, labels, info) in train_samples:
  result = await seq.step(inputs)

result = await seq.end(inputs)
```

Note that above example called `seq.end()` by sending the last inputs again to end the sequence. If you want to specify the inputs for the execution, you can run `result = await se.end(inputs)`.

For a small batch of data, you can also run it like this:
```python
from pyotritonclient import SequenceExcutor

seq = SequenceExcutor(
  server_url='https://ai.imjoy.io/triton',
  model_name='cellpose-train',
  sequence_id=100
)

# a list of inputs
inputs_batch = [[
  image.astype('float32'),
  labels.astype('float32'),
  {"steps": 1, "resume": True}
] for (image, labels, _) in train_samples]

def on_step(i, result):
  """Function called on every step"""
  print(i)

results = await seq(inputs_batch, on_step=on_step)
```



## Server setup
Since we access the server from the browser environment which typically has more security restrictions, it is important that the server is configured to enable browser access.

Please make sure you configured following aspects:
 * The server must provide HTTPS endpoints instead of HTTP
 * The server should send the following headers:
    - `Access-Control-Allow-Headers: Inference-Header-Content-Length,Accept-Encoding,Content-Encoding,Access-Control-Allow-Headers`
    - `Access-Control-Expose-Headers: Inference-Header-Content-Length,Range,Origin,Content-Type`
    - `Access-Control-Allow-Methods: GET,HEAD,OPTIONS,PUT,POST`
    - `Access-Control-Allow-Origin: *` (This is optional depending on whether you want to support CORS)
