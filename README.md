# Triton HTTP Client for Pyodide

A Pyodide python http client library and utilities for communicating with Triton Inference Server (based on tritonclient from NVIDIA).


This is a simplified implemetation of the triton client from NVIDIA, it is mainly made for running in the web browser with pyodide.
It only implement the http client, and most of the API remains the same but changed into async.


## Installation
To install it, run `pip install pyotritonclient`


## Usage
To use it in Pyodie, see the [test example](./tests/test_client.py).

You can also find the official [client examples](https://github.com/triton-inference-server/client/tree/main/src/python/examples) demonstrate how to use the 
package to issue request to [triton inference server](https://github.com/triton-inference-server/server). However, please notice that you will need to
change the http client code into async style. For example, instead of doing `client.infer(...)`, you need to do `await client.infer(...)`.

The http client code is forked from [triton client git repo](https://github.com/triton-inference-server/client) since commit [b3005f9db154247a4c792633e54f25f35ccadff0](https://github.com/triton-inference-server/client/tree/b3005f9db154247a4c792633e54f25f35ccadff0).


## Server setup
Since we will access the server from the browser environment which typically has more security restrictions, it is important that the server is configured to enable browser access.

Please make sure you configured following aspects:
 * The server must provide HTTPS endpoints instead of HTTP
 * The server should send the following headers:
    - `Access-Control-Allow-Headers: Inference-Header-Content-Length,Accept-Encoding,Content-Encoding,Access-Control-Allow-Headers`
    - `Access-Control-Expose-Headers: Inference-Header-Content-Length,Range,Origin,Content-Type`
    - `Access-Control-Allow-Methods: GET,HEAD,OPTIONS,PUT,POST`
    - `Access-Control-Allow-Origin: *` (This is optional depending on whether you want to support CORS)
