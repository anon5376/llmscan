"""Best-effort runtime BLAS thread cap without third-party dependencies.

Used only when multiple tensor workers run concurrently so nested BLAS threads do not
oversubscribe CPU cores. Only inspects shared libraries already loaded by the dynamic
linker (never data mmaps or unverified filesystem paths), leaves serial scans untouched,
never raises an existing lower BLAS limit, and degrades to a no-op on backends without a
runtime setter (such as Apple Accelerate).
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Callable

_LIB_MARKERS = ("openblas", "mkl_rt", "blis")
_BLAS_SYMBOLS = (
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
    ("openblas_get_num_threads", "openblas_set_num_threads"),
    ("MKL_Get_Max_Threads", "MKL_Set_Num_Threads"),
    ("bli_thread_get_num_threads", "bli_thread_set_num_threads"),
)

_RTLD_NOLOAD_LINUX = 4
_RTLD_NOLOAD_DARWIN = 0x10


def _matches_blas_lib(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not any(marker in name for marker in _LIB_MARKERS):
        return False
    if sys.platform.startswith("linux"):
        return name.endswith(".so") or ".so." in name
    if sys.platform == "darwin":
        return name.endswith(".dylib")
    if os.name == "nt":
        return name.endswith(".dll")
    return False


def _loaded_blas_paths() -> list[str]:
    """Paths of BLAS shared libraries already loaded by the dynamic linker."""
    paths: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        if p and p not in seen and _matches_blas_lib(p):
            seen.add(p)
            paths.append(p)

    if sys.platform.startswith("linux"):
        try:
            class _DlPhdrInfo(ctypes.Structure):
                _fields_ = [("dlpi_addr", ctypes.c_void_p), ("dlpi_name", ctypes.c_char_p)]

            cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_DlPhdrInfo), ctypes.c_size_t, ctypes.c_void_p)
            libc = ctypes.CDLL(None)
            libc.dl_iterate_phdr.argtypes = [cb_type, ctypes.c_void_p]
            libc.dl_iterate_phdr.restype = ctypes.c_int

            def _cb(info, _size, _data):
                raw = info.contents.dlpi_name
                if raw:
                    add(os.fsdecode(raw))
                return 0

            libc.dl_iterate_phdr(cb_type(_cb), None)
        except (OSError, AttributeError):
            pass
    elif sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None)
            libc._dyld_image_count.argtypes = []
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

    return paths


def _open_loaded_lib(path: str) -> ctypes.CDLL | None:
    """Open an already-loaded shared library without ever loading new native code from disk."""
    try:
        if sys.platform.startswith("linux"):
            return ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL | _RTLD_NOLOAD_LINUX)
        if sys.platform == "darwin":
            return ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL | _RTLD_NOLOAD_DARWIN)
        if os.name == "nt":
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("Kernel32.dll")
            kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            handle = kernel32.GetModuleHandleW(path)
            if not handle:
                return None
            return ctypes.CDLL(path, handle=handle)
    except OSError:
        return None
    return None


def _discover_controls() -> list[tuple[Callable[[], int], Callable[[int], None]]]:
    controls: list[tuple[Callable[[], int], Callable[[int], None]]] = []
    for path in _loaded_blas_paths():
        lib = _open_loaded_lib(path)
        if lib is None:
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
