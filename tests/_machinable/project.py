from machinable import Project

class TestEnv(Project):
    def on_resolve_remotes(self):
        return {
            "mpi": "url+https://raw.githubusercontent.com/machinable-org/machinable/2670e9626eb548f6ce2301923be1f49642086d8c/docs/examples/mpi-execution/mpi.py",
        }