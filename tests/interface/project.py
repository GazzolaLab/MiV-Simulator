from machinable import Project


class TestEnv(Project):
    def on_resolve_remotes(self):
        return {
            "mpi": "url+https://raw.githubusercontent.com/machinable-org/machinable/main/docs/examples/mpi-execution/mpi.py",
        }
