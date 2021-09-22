import json
from pyodide import to_js
from js import Object, fetch

class HTTPResponse():
    def __init__(self, js_response, body_buffer, method='GET'):
        self.method = method.upper()
        self.status_message = None
        self._js_response = js_response
        self._body_buffer = body_buffer

    def __getitem__(self, key):
        return self._js_response.headers.get(key)

    def get(self, key, default=None):
        return self._js_response.headers.get(key) or default

    async def read(self, length=None):
        # convert such a proxy to a Python memoryview using the to_py api
        # then convert to bytearray
        return self._body_buffer[:length].tobytes()

class pyohttpclient():
    def close(self):
        pass

    async def get(self, uri, headers=None):
        resp = await fetch(uri, to_js({
            "method": "GET",
            "headers": headers
        }, dict_converter=Object.fromEntries))
        _body_buffer = await resp.arrayBuffer()
        return HTTPResponse(resp, _body_buffer.to_py())

    async def post(self, uri, body=None, headers=None):
        if isinstance(body, dict):
            body = json.dumps(body)
        resp = await fetch(uri, to_js({
            "method": "POST",
            "body": body,
            "headers": headers
        }, dict_converter=Object.fromEntries))
        _body_buffer = await resp.arrayBuffer()
        return HTTPResponse(resp, _body_buffer.to_py())