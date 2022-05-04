# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
from setuptools import find_packages
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.startswith("#")]


install_requires = req_file("requirements.txt")
VERSION = json.loads(read("pyotritonclient/VERSION"))["version"]
# only for native python, not pyodide
install_requires.extend(
    [
        "requests;platform_system!='Emscripten'",
        "python-rapidjson>=0.9.1;platform_system!='Emscripten'",
        "msgpack",
    ]
)
data_files = [
    ("", ["LICENSE.txt"]),
]

setup(
    name="pyotritonclient",
    version=VERSION,
    author="Wei OUYANG",
    author_email="oeway007@gmail.com",
    description="A lightweight http client library for communicating with Nvidia Triton Inference Server (with Pyodide support in the browser)",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="BSD",
    url="https://github.com/oeway/pyotritonclient",
    keywords=[
        "pyodide",
        "http",
        "triton",
        "tensorrt",
        "inference",
        "server",
        "service",
        "client",
        "nvidia",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require={"imjoy": ["imjoy-rpc"]},
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    data_files=data_files,
)
