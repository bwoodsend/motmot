# -*- coding: utf-8 -*-
"""
Hook for PyInstaller.
"""

from motmot._slug import slug

# Put the cslug binary and its types json in a `motmot` directory.
datas = [(str(slug.path), "motmot"), (str(slug.types_map.json_path), "motmot")]
