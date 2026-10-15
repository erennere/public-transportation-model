"""Cached HTTP downloads."""
from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


def fetch(url: str, dest_dir: str | Path, name: str | None = None, timeout: int = 600) -> Path:
    """Download `url` into `dest_dir` unless already there. Returns the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / (name or Path(urlparse(url).path).name)
    if path.exists() and path.stat().st_size > 0:
        return path
    log.info("downloading %s", url)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=1 << 20)
    tmp.rename(path)
    return path


def unzip_if_needed(path: Path) -> list[Path]:
    if path.suffix.lower() != ".zip":
        return [path]
    with zipfile.ZipFile(path) as z:
        z.extractall(path.parent)
        return [path.parent / n for n in z.namelist()]
