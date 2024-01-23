import pytest
import os
import shutil

from miv_simulator.mechanisms import compile, compile_and_load


def test_mechanisms_compile(tmp_path):
    d = str(tmp_path / "mechanisms")
    os.makedirs(d)

    # invalid directory
    with pytest.raises(FileNotFoundError):
        compile()

    assert os.path.isdir(compile(d))
    assert os.path.exists(os.path.join(d, "compiled"))

    # skip if existing
    assert os.path.isdir(compile(d))
    # compile with -force
    compiled_path = compile(d, force=True)
    assert len(os.path.basename(compiled_path)) == 64


def test_mechanisms_compile_and_load(tmp_path):
    d = str(tmp_path / "mechanisms")
    os.makedirs(d)

    # Test if compile_and_load is called properly
    dll_path = compile_and_load(d)
    assert os.path.isfile(dll_path)

    # Test if compile_and_load creates the dll file
    os.remove(dll_path)
    assert not os.path.exists(dll_path)
    compile_and_load(d, force=True)
    assert os.path.exists(dll_path)
