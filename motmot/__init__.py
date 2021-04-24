# -*- coding: utf-8 -*-
"""
"""
from ._version import __version__, __version_info__
from . import geometry
from ._mesh import Mesh
from ._curvatures import Curvature
from . import connectivity


def _PyInstaller_hook_dir():
    import os
    return [os.path.dirname(__file__)]
