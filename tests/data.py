# -*- coding: utf-8 -*-
"""
"""

from pathlib import Path as _Path

HERE = _Path(__file__).parent.resolve()
DUMP_DIR = HERE / "_data"
DUMP_DIR.mkdir(exist_ok=True)


def _download_rabbit():
    path = DUMP_DIR / "rabbit.stl"
    if path.exists():
        return path

    url = "https://raw.githubusercontent.com/bwoodsend/vtkplotlib/master/" \
          "vtkplotlib/data/models/rabbit/rabbit.stl"

    from urllib import request
    with request.urlopen(url) as req:
        raw_bytes = req.read()

    path.write_bytes(raw_bytes)
    return path


rabbit_path = _download_rabbit()
