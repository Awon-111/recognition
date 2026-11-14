"""Start the local simulation service."""

import argparse
import tempfile

import uvicorn

from slag.config import RUNTIME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动铁水罐扒渣路径规划操作台")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    scratch = RUNTIME / "temp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    uvicorn.run("slag.api:app", host="127.0.0.1", port=args.port)
