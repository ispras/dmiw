#!/usr/bin/env python3
"""Download DMIW checkpoints and configurations from models/weights.json.

Python >= 3.10; standard library only. The default paths are relative to this
script's location, so the working directory does not matter. Copy scripts/ and
models/weights.json into the DMIW repository before running it.
"""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
BLOCK_SIZE = 1024 * 1024


def file_status(path: Path, item: dict) -> str:
    """Return ok only when both the byte length and the SHA-256 match."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size != item["size"]:
        return "size mismatch (possibly a Git LFS pointer)"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        return "SHA-256 mismatch"
    return "ok"


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("The manifest has no models")
    destinations = set()
    for name, model in models.items():
        if not isinstance(model, dict) or not model.get("files"):
            raise ValueError(f"Model {name!r} has no files")
        for item in model["files"]:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid file entry in model {name!r}")
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or "\\" in raw_path:
                raise ValueError(f"Invalid relative file path: {raw_path!r}")
            relative = PurePosixPath(raw_path)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
            ):
                raise ValueError(f"Invalid relative file path: {raw_path!r}")
            if relative.as_posix() in destinations:
                raise ValueError(f"Duplicate destination: {raw_path}")
            destinations.add(relative.as_posix())
            if type(item.get("size")) is not int or item["size"] <= 0:
                raise ValueError(f"Invalid byte length for {raw_path}")
            digest = item.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", digest
            ):
                raise ValueError(f"Invalid SHA-256 for {raw_path}")
            asset = item.get("asset")
            if (
                not isinstance(asset, str)
                or not asset
                or "/" in asset
                or "\\" in asset
            ):
                raise ValueError(f"Invalid asset name for {raw_path}")
    return data


def make_url(base_url: str | None, item: dict) -> str:
    if not base_url:
        raise ValueError("Set base_url in weights.json or supply --base-url")
    parts = urlsplit(base_url)
    if (
        parts.scheme != "https"
        or not parts.netloc
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "--base-url must be an HTTPS URL without query or fragment"
        )
    # A Nextcloud sharing page is HTML. Public WebDAV returns the actual file,
    # with paths relative to the shared models/ folder (no extra models/ prefix).
    share = re.fullmatch(
        r"(.*?)(?:/index\.php)?/s/([^/]+)", parts.path.rstrip("/")
    )
    if share:
        prefix, token = share.groups()
        dav_path = (
            prefix + "/public.php/dav/files/" + quote(unquote(token), safe="")
        )
        base_url = urlunsplit((parts.scheme, parts.netloc, dav_path, "", ""))
        return base_url + "/" + quote(item["path"], safe="/")
    if re.search(r"/public\.php/dav/files/[^/]+/?$", parts.path):
        return base_url.rstrip("/") + "/" + quote(item["path"], safe="/")
    # Preserve support for flat HTTP directories, including GitHub Releases.
    return base_url.rstrip("/") + "/" + quote(item["asset"], safe="")


def target_path(destination: Path, item: dict) -> Path:
    target = (destination / item["path"]).resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"File path escapes destination: {item['path']}")
    return target


def download_one(url: str, target: Path, item: dict, timeout: float) -> None:
    """Stream to a unique temp file, verify, then atomically replace target.

    A failed transfer preserves the existing target and deletes its temporary
    file. Retries restart that file's download; HTTP Range resume is not used.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".part", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            request = Request(
                url,
                headers={
                    "User-Agent": "dmiw-model-downloader/1",
                    "Accept-Encoding": "identity",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Refusing a redirect to a non-HTTPS URL")
                if response.status != 200:
                    raise ValueError(
                        f"Unexpected HTTP status: {response.status}"
                    )
                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if content_type in ("text/html", "application/xhtml+xml"):
                    raise ValueError(
                        "Server returned an HTML page instead of a model file; check the share link and download permissions"
                    )
                content_length = response.headers.get("Content-Length")
                if (
                    content_length is not None
                    and int(content_length) != item["size"]
                ):
                    raise ValueError(
                        "Server file length does not match the manifest"
                    )
                digest = hashlib.sha256()
                received = 0
                last_report = time.monotonic()
                while block := response.read(BLOCK_SIZE):
                    received += len(block)
                    if received > item["size"]:
                        raise ValueError(
                            "Server sent more bytes than the manifest allows"
                        )
                    digest.update(block)
                    output.write(block)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        print(
                            f"  {received / item['size']:.0%} ({received / 1e6:.1f} MB)",
                            flush=True,
                        )
                        last_report = now
                if received != item["size"]:
                    raise ValueError(
                        f"Incomplete download: {received}/{item['size']} bytes"
                    )
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(
                        "Downloaded file SHA-256 does not match the manifest"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "models" / "weights.json"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "models",
        help="root containing algorithm directories",
    )
    parser.add_argument(
        "--base-url",
        help="override the Nextcloud public share link or the HTTP download root from the manifest",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model", nargs="+", metavar="NAME", help="model names from --list"
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="select all checkpoints (about 4.37 GB)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list models without downloading"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only check local files; selects all if no names supplied",
    )
    parser.add_argument("--attempts", type=positive_int, default=3)
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=60,
        help="socket timeout in seconds",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        models = manifest["models"]
        names = (
            list(models)
            if args.all or (args.verify and not args.model)
            else list(dict.fromkeys(args.model or []))
        )
        unknown = set(names) - set(models)
        if unknown:
            parser.error("Unknown model(s): " + ", ".join(sorted(unknown)))
        if args.list:
            for name, model in models.items():
                size = sum(item["size"] for item in model["files"]) / 1e6
                print(
                    f"{name:20} {size:9.2f} MB  {model['algorithm']} --model {model['variant']}"
                )
            return 0
        if not names:
            parser.error("Select --model NAME, --all, --list or --verify")
        destination = args.dest.resolve()
        base_url = args.base_url or manifest.get("base_url")
        selected = [item for name in names for item in models[name]["files"]]
        # Validate every path and required URL before downloading any file.
        work = []
        failures = 0
        for item in selected:
            target = target_path(destination, item)
            status = file_status(target, item)
            if args.verify:
                print(f"{item['path']}: {status}")
                failures += status != "ok"
            elif status == "ok":
                print(f"SKIP {item['path']} (verified)")
            else:
                work.append((item, target, make_url(base_url, item)))
        for item, target, url in work:
            print(
                f"GET {item['path']} ({item['size'] / 1e6:.2f} MB)", flush=True
            )
            for attempt in range(1, args.attempts + 1):
                try:
                    download_one(url, target, item, args.timeout)
                except (OSError, HTTPException, ValueError) as exc:
                    print(
                        f"  attempt {attempt}/{args.attempts}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if isinstance(exc, HTTPError) and exc.code in (
                        400,
                        401,
                        403,
                        404,
                    ):
                        failures += 1
                        break
                    if attempt == args.attempts:
                        failures += 1
                    else:
                        time.sleep(min(attempt, 3))
                else:
                    print(
                        f"OK {item['path']} (size and SHA-256 verified)",
                        flush=True,
                    )
                    break
        return 1 if failures else 0
    except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nDownload interrupted; existing files were preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
