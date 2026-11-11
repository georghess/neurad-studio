"""Build the viewer once and discard its Node build tools from container images."""

import shutil
from pathlib import Path

import viser

if __name__ == "__main__":
    viser.ViserServer(host="127.0.0.1", port=0).stop()
    client = Path(viser.__file__).parent / "client"
    if not (client / "build/index.html").is_file():
        raise RuntimeError("Viser frontend build failed")
    for name in ("node_modules", ".nodeenv"):
        path = client / name
        if path.exists():
            shutil.rmtree(path)
