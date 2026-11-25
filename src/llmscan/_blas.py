"""Best-effort runtime BLAS thread cap without third-party dependencies.

Used only when multiple tensor workers run concurrently so nested BLAS threads do not
oversubscribe CPU cores. Leaves single-threaded / serial scans untouched, never raises an
existing lower BLAS limit, and degrades to a no-op on backends without a runtime setter
(such as Apple Accelerate).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_LIB_MARKERS = ("openblas", "mkl_rt", "blis")
_BLAS_SYMBOLS = (
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
    ("openblas_get_num_threads", "openblas_set_num_threads"),
    ("MKL_Get_Max_Threads", "MKL_Set_Num_Threads"),
    ("bli_thread_get_num_threads", "bli_thread_set_num_threads"),
)


def _matches_blas(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(marker in name for marker in _LIB_MARKERS)


def _loaded_blas_paths() -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        if p and p not in seen and _matches_blas(p) and os.path.isfile(p):
            seen.add(p)
            paths.append(p)

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        add(parts[5])
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None)
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            for i in range(libc._dyld_image_count()):
                raw = libc._dyld_get_image_name(i)
                if raw:
                    add(os.fsdecode(raw))
        except (OSError, AttributeError):
            pass
    elif os.name == "nt":
        try:
            from ctypes import wintypes

            psapi = ctypes.WinDLL("Psapi.dll")
            kernel32 = ctypes.WinDLL("Kernel32.dll")
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.EnumProcessModules.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
            ]
            psapi.EnumProcessModules.restype = wintypes.BOOL
            psapi.GetModuleFileNameExW.argtypes = [
                wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD,
            ]
            psapi.GetModuleFileNameExW.restype = wintypes.DWORD
            hproc = kernel32.GetCurrentProcess()
            needed = wintypes.DWORD()
            buf_len = 1024
            mods = (wintypes.HMODULE * buf_len)()
            if psapi.EnumProcessModules(hproc, mods, ctypes.sizeof(mods), ctypes.byref(needed)):
                count = min(buf_len, needed.value // ctypes.sizeof(wintypes.HMODULE))
                name_buf = ctypes.create_unicode_buffer(1024)
                for i in range(count):
                    if psapi.GetModuleFileNameExW(hproc, mods[i], name_buf, len(name_buf)):
                        add(name_buf.value)
        except (OSError, AttributeError):
            pass

    np_dir = Path(np.__file__).resolve().parent
    for lib_dir in (np_dir / ".dylibs", np_dir.parent / "numpy.libs"):
        if lib_dir.is_dir():
            for entry in sorted(lib_dir.iterdir()):
                add(str(entry))

    return paths


def _discover_controls() -> list[tuple[Callable[[], int], Callable[[int], None]]]:
    controls: list[tuple[Callable[[], int], Callable[[int], None]]] = []
    for path in _loaded_blas_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for get_name, set_name in _BLAS_SYMBOLS:
            get_fn = getattr(lib, get_name, None)
            set_fn = getattr(lib, set_name, None)
            if get_fn is not None and set_fn is not None:
                get_fn.argtypes = []
                get_fn.restype = ctypes.c_int
                set_fn.argtypes = [ctypes.c_int]
                set_fn.restype = None
                controls.append((get_fn, set_fn))
                break
    return controls


class BlasThreadCap:
    """Temporarily cap BLAS threads across loaded libraries when running parallel workers."""

    def __init__(self, workers: int) -> None:
        self._workers = max(1, workers)
        self._saved: list[tuple[Callable[[int], None], int]] = []

    def apply(self) -> None:
        if self._workers <= 1:
            return
        for get_fn, set_fn in _discover_controls():
            try:
                prev = int(get_fn())
            except Exception:
                continue
            if prev <= 1:
                continue
            target = max(1, prev // self._workers)
            if target < prev:
                try:
                    set_fn(target)
                    self._saved.append((set_fn, prev))
                except Exception:
                    continue

    def restore(self) -> None:
        while self._saved:
            set_fn, prev = self._saved.pop()
            try:
                set_fn(prev)
            except Exception:
                pass
