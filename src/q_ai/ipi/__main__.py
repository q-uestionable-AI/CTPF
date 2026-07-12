"""Allow ``python -m q_ai.ipi`` to run the IPI Typer app.

Used by the managed-listener subprocess so IPI listen works after the
root CLI no longer registers the ``ipi`` subcommand.
"""

from q_ai.ipi.cli import app

app()
