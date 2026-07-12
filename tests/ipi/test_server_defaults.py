import inspect
import unittest

from typer.testing import CliRunner

from q_ai.ipi.cli import app
from q_ai.ipi.commands.listen import _is_loopback_host, listen
from q_ai.ipi.server import start_server


class TestServerDefaults(unittest.TestCase):
    def test_start_server_defaults(self):
        """Verify that start_server defaults to 127.0.0.1."""
        sig = inspect.signature(start_server)
        params = sig.parameters
        self.assertEqual(params["host"].default, "127.0.0.1")
        self.assertEqual(params["port"].default, 8080)

    def test_cli_listen_defaults(self):
        """Verify that CLI listen command defaults to 127.0.0.1."""
        sig = inspect.signature(listen)
        params = sig.parameters
        self.assertEqual(params["host"].default, "127.0.0.1")
        self.assertEqual(params["port"].default, 8080)

    def test_loopback_host_helpers(self):
        """Accept loopback names/addresses; reject external binds."""
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertTrue(_is_loopback_host("::1"))
        self.assertFalse(_is_loopback_host("0.0.0.0"))
        self.assertFalse(_is_loopback_host("192.168.1.10"))

    def test_cli_rejects_non_loopback_host(self):
        """``qai ipi listen --host 0.0.0.0`` must exit non-zero."""
        runner = CliRunner()
        result = runner.invoke(app, ["listen", "--host", "0.0.0.0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("non-loopback", result.output.lower())

    def test_cli_rejects_tunnel_with_mismatched_loopback_host(self):
        """Tunnel mode must not bind a loopback address cloudflared won't hit."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["listen", "--host", "127.0.0.2", "--tunnel", "cloudflare"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("127.0.0.1 or localhost", result.output)


if __name__ == "__main__":
    unittest.main()
