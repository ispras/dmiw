"""Downloader checks with simulated HTTPS responses; no model dependencies."""

import contextlib
import hashlib
from http.client import IncompleteRead
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = (
            {} if length is None else {"Content-Length": str(length)}
        )

    def geturl(self):
        return "https://weights.example.test/model.ckpt"


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b"example checkpoint binary content\x00\xff"
        self.item = {
            "path": "trustmark/encoder_P.ckpt",
            "asset": "encoder_P.ckpt",
            "size": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.target = self.root / self.item["path"]
        self.manifest = self.root / "weights.json"
        self.data = {
            "schema_version": 1,
            "base_url": "https://weights.example.test/models-v1",
            "models": {
                "test": {
                    "algorithm": "trustmark",
                    "variant": "P",
                    "files": [self.item],
                }
            },
        }
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def run_cli(self, *args):
        with contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            return dl.main(
                [
                    "--manifest",
                    str(self.manifest),
                    "--dest",
                    str(self.root),
                    *args,
                ]
            )

    def test_valid_download_replaces_lfs_pointer(self):
        self.target.parent.mkdir()
        self.target.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
        )
        with patch.object(
            dl,
            "urlopen",
            return_value=Response(self.payload, len(self.payload)),
        ):
            self.assertEqual(self.run_cli("--model", "test"), 0)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_verified_checkpoint_skips_network(self):
        self.target.parent.mkdir()
        self.target.write_bytes(self.payload)
        with patch.object(
            dl,
            "urlopen",
            side_effect=AssertionError("Network must not be used"),
        ):
            self.assertEqual(self.run_cli("--model", "test"), 0)

    def test_bad_sha_does_not_replace_existing_file(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"previous checkpoint")
        wrong = b"x" * len(self.payload)
        with patch.object(dl, "urlopen", return_value=Response(wrong)):
            self.assertEqual(
                self.run_cli("--model", "test", "--attempts", "1"), 1
            )
        self.assertEqual(self.target.read_bytes(), b"previous checkpoint")
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_truncated_download_is_removed(self):
        with patch.object(
            dl, "urlopen", return_value=Response(self.payload[:-1])
        ):
            self.assertEqual(
                self.run_cli("--model", "test", "--attempts", "1"), 1
            )
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_oversized_response_is_rejected(self):
        with patch.object(
            dl, "urlopen", return_value=Response(self.payload + b"x")
        ):
            self.assertEqual(
                self.run_cli("--model", "test", "--attempts", "1"), 1
            )
        self.assertFalse(self.target.exists())

    def test_wrong_content_length_is_rejected(self):
        with patch.object(
            dl,
            "urlopen",
            return_value=Response(self.payload, len(self.payload) + 1),
        ):
            self.assertEqual(
                self.run_cli("--model", "test", "--attempts", "1"), 1
            )
        self.assertFalse(self.target.exists())

    def test_failed_connection_is_retried(self):
        with patch.object(
            dl,
            "urlopen",
            side_effect=[OSError("Connection reset"), Response(self.payload)],
        ) as network:
            with patch.object(dl.time, "sleep"):
                self.assertEqual(
                    self.run_cli("--model", "test", "--attempts", "2"), 0
                )
        self.assertEqual(network.call_count, 2)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_broken_chunked_response_is_reported_and_cleaned_up(self):
        response = Response(self.payload)
        with patch.object(
            response, "read", side_effect=IncompleteRead(b"partial", 10)
        ):
            with patch.object(dl, "urlopen", return_value=response):
                self.assertEqual(
                    self.run_cli("--model", "test", "--attempts", "1"), 1
                )
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_verify_reports_missing_files_without_network(self):
        with patch.object(
            dl,
            "urlopen",
            side_effect=AssertionError("No downloads in verify mode"),
        ):
            self.assertEqual(self.run_cli("--verify"), 1)

    def test_verify_checks_same_size_corruption(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"x" * len(self.payload))
        self.assertEqual(self.run_cli("--verify"), 1)

    def test_listing_needs_no_url(self):
        self.data["base_url"] = None
        self.write_manifest()
        with patch.object(
            dl,
            "urlopen",
            side_effect=AssertionError("No downloads in list mode"),
        ):
            self.assertEqual(self.run_cli("--list"), 0)

    def test_missing_url_fails_before_creating_model_directories(self):
        self.data["base_url"] = None
        self.write_manifest()
        self.assertEqual(self.run_cli("--model", "test"), 1)
        self.assertFalse(self.target.parent.exists())

    def test_manifest_path_traversal_is_rejected(self):
        self.item["path"] = "../outside.ckpt"
        self.write_manifest()
        with self.assertRaises(ValueError):
            dl.load_manifest(self.manifest)

    def test_url_quotes_asset_names(self):
        item = dict(self.item, asset="model name.ckpt")
        self.assertEqual(
            dl.make_url("https://weights.example.test/v1", item),
            "https://weights.example.test/v1/model%20name.ckpt",
        )

    def test_nextcloud_share_page_becomes_direct_file_url(self):
        url = dl.make_url(
            "https://cloud.example.test/index.php/s/SHARE", self.item
        )
        self.assertEqual(
            url,
            "https://cloud.example.test/public.php/dav/files/SHARE/trustmark/encoder_P.ckpt",
        )

    def test_nextcloud_pretty_share_url(self):
        url = dl.make_url("https://cloud.example.test/s/SHARE/", self.item)
        self.assertEqual(
            url,
            "https://cloud.example.test/public.php/dav/files/SHARE/trustmark/encoder_P.ckpt",
        )

    def test_nextcloud_installation_in_subdirectory(self):
        url = dl.make_url(
            "https://cloud.example.test/cloud/index.php/s/SHARE", self.item
        )
        self.assertEqual(
            url,
            "https://cloud.example.test/cloud/public.php/dav/files/SHARE/trustmark/encoder_P.ckpt",
        )

    def test_nextcloud_direct_dav_root_preserves_algorithm_directory(self):
        url = dl.make_url(
            "https://cloud.example.test/public.php/dav/files/SHARE/", self.item
        )
        self.assertEqual(
            url,
            "https://cloud.example.test/public.php/dav/files/SHARE/trustmark/encoder_P.ckpt",
        )

    def test_nextcloud_path_with_spaces_is_encoded(self):
        item = dict(self.item, path="my models/encoder P.ckpt")
        url = dl.make_url("https://cloud.example.test/index.php/s/SHARE", item)
        self.assertEqual(
            url,
            "https://cloud.example.test/public.php/dav/files/SHARE/my%20models/encoder%20P.ckpt",
        )

    def test_manifest_nextcloud_source_downloads_without_url_argument(self):
        self.data["base_url"] = "https://cloud.example.test/index.php/s/SHARE"
        self.write_manifest()
        with patch.object(
            dl, "urlopen", return_value=Response(self.payload)
        ) as network:
            self.assertEqual(self.run_cli("--model", "test"), 0)
        self.assertEqual(
            network.call_args.args[0].full_url,
            "https://cloud.example.test/public.php/dav/files/SHARE/trustmark/encoder_P.ckpt",
        )

    def test_html_share_page_is_rejected(self):
        response = Response(self.payload)
        response.headers["Content-Type"] = "text/html; charset=UTF-8"
        with patch.object(dl, "urlopen", return_value=response):
            self.assertEqual(
                self.run_cli("--model", "test", "--attempts", "1"), 1
            )
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_interrupt_preserves_existing_target(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"previous checkpoint")
        with patch.object(dl, "urlopen", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                dl.download_one(
                    "https://weights.example.test/a", self.target, self.item, 1
                )
        self.assertEqual(self.target.read_bytes(), b"previous checkpoint")
        self.assertFalse(list(self.root.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()
