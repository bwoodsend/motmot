import sys
import os
import runpy
import io

import pytest

from motmot._compat import cached_property
from motmot._misc import Independency, open_

from tests.data import HERE

pytestmark = pytest.mark.order(0)


def test():
    independent = Independency()

    class Foo(object):
        x = 0
        y = 0

        def set_x(self, new):
            self.x = new
            reset_on_set_x(self)

        def set_y(self, new):
            self.y = new
            reset_on_set_y(self)

        @independent.of("set_y")
        @cached_property
        def x_mirror(self):
            return self.x

        @independent.of("set_x")
        @cached_property
        def y_mirror(self):
            return self.y

        @cached_property
        def total(self):
            return self.x + self.y

    independent.init(Foo)
    reset_on_set_x = independent.reset_on("set_x")
    reset_on_set_y = independent.reset_on("set_y")

    self = Foo()
    self.x, self.y = 1, 2
    # All cached properties should initialise to the correct values.
    assert self.x_mirror == 1
    assert self.y_mirror == 2
    assert self.total == 3

    self.x, self.y = (3, 4)
    # The cached properties should not have been reset and still hold out of
    # date values.
    assert self.x_mirror == 1
    assert self.y_mirror == 2
    assert self.total == 3

    self.set_x(5)
    # set_x() invalidates the cache on x_mirror and total but not y_mirror.
    assert self.x_mirror == 5
    assert self.y_mirror == 2
    assert self.total == 9

    self.x = 6
    self.set_y(7)
    # set_y() invalidates the cache on y_mirror and total but not x_mirror.
    assert self.x_mirror == 5
    assert self.y_mirror == 7
    assert self.total == 13

    self.x, self.y = 8, 9
    # reset_all() should reset everything.
    independent.reset_all(self)
    assert self.x_mirror == 8
    assert self.y_mirror == 9
    assert self.total == 17


def test_PyInstaller_hook():
    if getattr(sys, "frozen", False):
        from motmot._slug import slug
        assert slug.path.exists()
        assert slug.types_map.json_path.exists()

    else:
        from motmot import _PyInstaller_hook_dir
        hook_dir, = _PyInstaller_hook_dir()
        assert os.path.isdir(hook_dir)
        hook = os.path.join(hook_dir, "hook-motmot.py")
        assert os.path.isfile(hook)

        namespace = runpy.run_path(hook)
        assert len(namespace["datas"]) == 2


compressed_files = list((HERE / "compressed_files").glob("*"))


@pytest.mark.parametrize("path", compressed_files,
                         ids=[i.stem for i in compressed_files])
def test_read_archived(path):
    """Decompress each file in tests/compressed_files and check that they say
     what they're supposed to say."""
    with open_(path, "rb") as f:
        contents = f.read().strip()
    # Each file contains: "This file is compression-method compressed."
    assert contents == f"This file is {path.stem} compressed.".encode()


def test_open_pipe():
    """Test opening a raw file handle number."""
    read, write = os.pipe()
    with open_(write, "wb") as f:
        f.write(b"hello\n")

    with open_(read, "rb") as f:
        assert f.read() == b"hello\n"


def test_open_buffer():
    """Test opening an already open buffer."""
    write = io.BytesIO()
    with open_(write, "wb"):
        write.write(b"hello\n")
    assert not write.closed
    assert write.getvalue() == b"hello\n"
