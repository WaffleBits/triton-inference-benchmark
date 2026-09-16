from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coordinated_benchmark import _parse_args
from remote_agent import (
    _parse_args as parse_agent_args,
    validate_agent_ca_file,
    validate_tls_material_file,
)


class TlsTransportTest(unittest.TestCase):
    def test_ca_file_must_be_a_regular_non_symlink_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ca_file = root / "ca.pem"
            ca_file.write_text("certificate", encoding="utf-8")
            self.assertEqual(validate_agent_ca_file(ca_file), ca_file)

            directory = root / "directory"
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                validate_agent_ca_file(directory)

            symlink = root / "ca-link.pem"
            symlink.symlink_to(ca_file)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                validate_agent_ca_file(symlink)

    def test_server_tls_material_rejects_missing_or_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            certificate = root / "certificate.pem"
            certificate.write_text("certificate", encoding="utf-8")
            key = root / "key.pem"
            key.write_text("key", encoding="utf-8")
            self.assertEqual(validate_tls_material_file(certificate, "certificate"), certificate)
            self.assertEqual(validate_tls_material_file(key, "key"), key)

            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_tls_material_file(root / "missing.pem", "certificate")

            symlink = root / "key-link.pem"
            symlink.symlink_to(key)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                validate_tls_material_file(symlink, "key")

    def test_coordinator_requires_explicit_ca_file_with_remote_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ca_file = Path(temp_dir) / "ca.pem"
            ca_file.write_text("certificate", encoding="utf-8")
            with patch.dict(os.environ, {"TLS_TEST_AGENT_KEY": "agent-key-32-characters-long"}):
                with patch(
                    "sys.argv",
                    [
                        "coordinated_benchmark.py",
                        "--clients",
                        "2",
                        "--output-dir",
                        temp_dir,
                        "--agent-url",
                        "https://agent-a.example.test",
                        "--agent-url",
                        "https://agent-b.example.test",
                        "--agent-api-key-env",
                        "TLS_TEST_AGENT_KEY",
                        "--agent-ca-file",
                        str(ca_file),
                        "--",
                        "--mode",
                        "mock",
                        "--num-requests",
                        "2",
                    ],
                ):
                    args = _parse_args()

            self.assertEqual(args.agent_ca_file, ca_file)

            with patch(
                "sys.argv",
                [
                    "coordinated_benchmark.py",
                    "--clients",
                    "2",
                    "--output-dir",
                    temp_dir,
                    "--agent-ca-file",
                    str(ca_file),
                ],
            ):
                with self.assertRaises(SystemExit):
                    _parse_args()

    def test_agent_requires_tls_certificate_and_key_as_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            certificate = root / "certificate.pem"
            certificate.write_text("certificate", encoding="utf-8")
            with patch.dict(os.environ, {"TLS_TEST_AGENT_KEY": "agent-key-32-characters-long"}):
                with patch(
                    "sys.argv",
                    [
                        "remote_agent.py",
                        "--agent-id",
                        "tls-agent",
                        "--api-key-env",
                        "TLS_TEST_AGENT_KEY",
                        "--tls-cert-file",
                        str(certificate),
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        parse_agent_args()


if __name__ == "__main__":
    unittest.main()
