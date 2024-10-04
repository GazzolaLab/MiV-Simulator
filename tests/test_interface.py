import pytest
import os


@pytest.mark.skipif(
    not os.environ.get("MIV_SIMULATOR_TEST_INTERFACE_SRC"),
    reason="Environment variables not available.",
)
def test_interface(tmp_path):
    from machinable import get
    from miv_simulator.mechanisms import compile

    source = os.environ["MIV_SIMULATOR_TEST_INTERFACE_SRC"]
    wd = os.path.dirname(__file__)

    debug = True
    storage_directory = str(tmp_path / "storage")
    if debug:
        storage_directory = f"{wd}/interface/storage"

    with get("machinable.index", storage_directory), get("machinable.project", wd):
        with get("mpi") as run:
            get(
                "miv_simulator.interface.network",
                {
                    "config_filepath": f"{source}/config/Microcircuit.yaml",
                    "mechanisms_path": compile(f"{source}/mechanisms"),
                    "template_path": f"{source}/templates",
                    "morphology_path": f"{source}/morphology",
                },
            ).launch()

        for component in run.executables:
            print(component)
            assert component.cached()
