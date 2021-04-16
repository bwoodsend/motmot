# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with motmot included.
"""
import sys
import pytest
import motmot

pytest.main(sys.argv[1:] + ["--no-cov", "--tb=native"])
