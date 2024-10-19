"""
Freeze pytest.main() with motmot included.
"""
import sys
import pytest
import motmot

status = pytest.main(sys.argv[1:] + ["--no-cov", "--tb=native"])
assert status.value == 0
