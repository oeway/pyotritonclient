python build_wheel.py --dest-dir ./
VERSION=0.1.0 pip install ./wheel[http]
cd tests && python tests.py