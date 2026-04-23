import pytest
import os

from miv_simulator.mechanisms import compile, compile_and_load


def test_mechanisms_compile(tmp_path):
    d = str(tmp_path / "mechanisms")
    os.makedirs(d)

    with open("tests/mechanisms/Gfluct3.mod", "r") as f:
        with open(os.path.join(d, "Gfluct3.mod"), "w") as g:
            g.write(f.read())

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

    h1 = compile(d, force=True, return_hash=True)
    h2 = compile(d, force=True, return_hash=True)
    assert h1 == h2


def test_mechanisms_compile_and_load(tmp_path):
    d = str(tmp_path / "mechanisms")
    os.makedirs(d)

    with open("tests/mechanisms/Gfluct3.mod", "r") as f:
        with open(os.path.join(d, "Gfluct3.mod"), "w") as g:
            g.write(f.read())

    # Test if compile_and_load is called properly
    try:
        dll_path = compile_and_load(d)
    except RuntimeError:
        # NEURON does not allow loading the same mechanism symbols from a
        # different path; skip if another test already loaded them.
        pytest.skip("Mechanisms already loaded from another path")
    assert os.path.isdir(dll_path)

    # Test if compile_and_load creates the dll directory
    compile_and_load(d, force=True)
    assert os.path.isdir(dll_path)
