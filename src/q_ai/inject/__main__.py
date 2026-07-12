"""Allow ``python -m q_ai.inject`` to run the inject fixture Typer app.

Inject is not registered on the root ``qai`` CLI after CTPF reconnect;
this module entry point is the documented way to serve and list fixtures.
"""

from q_ai.inject.cli import app

if __name__ == "__main__":
    app()
