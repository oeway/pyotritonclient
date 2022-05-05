import json
import requests
import asyncio
import functools


class HTTPResponse:
    def __init__(self, response, body_buffer, method="GET"):
        self.method = method.upper()
        self.status_message = None
        self.status_code = response.status_code
        self.msg = response.reason
        self._response = response
        self._body_buffer = body_buffer
        self._buffer_pointer = 0
        self.expose_headers = self._response.headers.get(
            "Access-Control-Expose-Headers"
        )

    def __getitem__(self, key):
        return self._response.headers.get(key)

    def get(self, key, default=None):
        if "content-encoding" == key.lower():
            if self.expose_headers is None or (
                "Content-Encoding" not in self.expose_headers
                and "content-encoding" not in self.expose_headers
            ):
                return None
        return self._response.headers.get(key) or default

    def read(self, length=None):
        if self._buffer_pointer >= len(self._body_buffer):
            raise EOFError
        end = None if length is None else self._buffer_pointer + length
        # convert such a proxy to a Python memoryview using the to_py api
        # then convert to bytearray
        ret = self._body_buffer[self._buffer_pointer : end]
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
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, functools.partial(requests.get, request_uri, headers=headers))
        # resp = requests.get(request_uri, headers=headers)
        _body_buffer = resp.content
        return HTTPResponse(resp, _body_buffer)

    async def post(self, request_uri, body=None, headers=None):
        request_uri = self._normalize_uri(request_uri)
        if isinstance(body, dict):
            body = json.dumps(body)
        elif isinstance(body, bytes):
            pass
        elif isinstance(body, str):
            raise TypeError("Unsupported type: " + str(type(body)))
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, functools.partial(requests.post, request_uri, data=body, headers=headers))
        # resp = requests.post(request_uri, data=body, headers=headers)
        _body_buffer = resp.content
        return HTTPResponse(resp, _body_buffer)
