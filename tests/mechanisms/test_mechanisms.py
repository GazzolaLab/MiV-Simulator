import pytest
import os
import commandlib

from miv_simulator.mechanisms import compile, compile_and_load


def test_mechanisms_compile():
    # compile with -force
    compiled_path = compile(force=True)
    assert os.path.basename(compiled_path) == "compiled"

    # calling compile() after initial compilation
    compile()


def test_mechanisms_compile_and_load():
    # Test if compile_and_load is called properly
    dll_path = compile_and_load()
    assert os.path.exists(dll_path)

    # Test if compile_and_load creates the dll file
    remove_cmd = commandlib.Command("rm", "-f")
    remove_cmd(dll_path).run()
    assert not os.path.exists(dll_path)
    compile_and_load(force=True)
    assert os.path.exists(dll_path)
