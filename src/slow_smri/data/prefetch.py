import logging
import secrets
import tempfile
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import huggingface_hub.utils
import fsspec
from fsspec.implementations.local import LocalFileSystem

huggingface_hub.utils.disable_progress_bars()

logger = logging.getLogger(__name__)


def prefetch(
    root: str,
    paths: list[str],
    *,
    cache_dir: str | Path | None = None,
    max_workers: int = 1,
    storage_options: dict | None = None,
):
    delete = cache_dir is None
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="prefetch-")

    fs: fsspec.AbstractFileSystem
    fs, root_ = fsspec.url_to_fs(root, **(storage_options or {}))
    is_remote = not isinstance(fs, LocalFileSystem)

    def fn(path: str):
        full_path = f"{root_}/{path}"
        if is_remote:
            tmp_path = Path(cache_dir) / full_path.lstrip("/")
            if not tmp_path.exists():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                get_file_atomic(fs, full_path, tmp_path)
            full_path = str(tmp_path)
        return path, full_path

    try:
        with ThreadPoolExecutor(max_workers) as executor:
            for future in buffer_map(executor, fn, paths, buffersize=2 * max_workers):
                try:
                    path, full_path = future.result()
                except Exception as e:
                    logger.warning("prefetch failed, skipping: %s", e)
                    continue
                yield path, full_path

                if delete and is_remote:
                    Path(full_path).unlink(missing_ok=True)
    finally:
        if delete:
            shutil.rmtree(cache_dir)


def buffer_map(
    executor: ThreadPoolExecutor,
    fn: Callable,
    *iterables: Iterable,
    buffersize: int = 16,
):
    window = deque()
    it = iter(zip(*iterables))

    for _ in range(buffersize):
        args = next(it, None)
        if args is None:
            break
        window.append(executor.submit(fn, *args))

    while window:
        future = window.popleft()
        yield future

        args = next(it, None)
        if args is not None:
            window.append(executor.submit(fn, *args))


def get_file_atomic(fs: fsspec.AbstractFileSystem, rpath: str, lpath: str | Path, *args, **kwargs):
    lpath = Path(lpath)
    staging = lpath.with_name(f".tmp-{secrets.token_hex(3)}-{lpath.name}")
    try:
        fs.get_file(rpath, str(staging), *args, **kwargs)
        staging.rename(lpath)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
