import json
from pyodide import to_js
from js import Object, fetch


class HTTPResponse:
    def __init__(self, js_response, body_buffer, method="GET"):
        self.method = method.upper()
        self.status_message = None
        self.status_code = js_response.status
        self._js_response = js_response
        self._body_buffer = body_buffer
        self._buffer_pointer = 0
        self.expose_headers = self._js_response.headers.get(
            "Access-Control-Expose-Headers"
        )

    def __getitem__(self, key):
        return self._js_response.headers.get(key)

    def get(self, key, default=None):
        if "content-encoding" == key.lower():
            if self.expose_headers is None or (
                "Content-Encoding" not in self.expose_headers
                and "content-encoding" not in self.expose_headers
            ):
                return None
        return self._js_response.headers.get(key) or default

    def read(self, length=None):

        if self._buffer_pointer >= len(self._body_buffer):
            raise EOFError
        end = None if length is None else self._buffer_pointer + length
        # convert such a proxy to a Python memoryview using the to_py api
        # then convert to bytearray
        ret = self._body_buffer[self._buffer_pointer : end].tobytes()
        self._buffer_pointer += len(ret)
        return ret


class HttpClient:
    def __init__(self, base_uri):
        self._base_uri = base_uri
        if not self._base_uri.startswith("http"):
            raise Exception("Please configure base URI for the http client.")

    def close(self):
        pass

    def _normalize_uri(self, request_uri):
        if not request_uri.startswith("http"):
            request_uri = self._base_uri.rstrip("/") + "/" + request_uri.lstrip("/")
        return request_uri

    async def get(self, request_uri, headers=None):
        request_uri = self._normalize_uri(request_uri)
        resp = await fetch(
            request_uri,
            to_js(
                {"method": "GET", "headers": headers}, dict_converter=Object.fromEntries
            ),
        )
        _body_buffer = await resp.arrayBuffer()
        return HTTPResponse(resp, _body_buffer.to_py())

    async def post(self, request_uri, body=None, headers=None):
        request_uri = self._normalize_uri(request_uri)
        if isinstance(body, dict):
            body = json.dumps(body)
        elif isinstance(body, bytes):
            body = to_js(body)  # this is Uint8Array
        elif isinstance(body, str):
            raise TypeError("Unsupported type: " + str(type(body)))

        resp = await fetch(
            request_uri,
            to_js(
                {"method": "POST", "body": body, "headers": headers},
                dict_converter=Object.fromEntries,
            ),
        )
        _body_buffer = await resp.arrayBuffer()
        return HTTPResponse(resp, _body_buffer.to_py())
