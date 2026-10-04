# ==============================================================================
# Project PULSAR-ASM | runtime/pulsar_abi.py
# ------------------------------------------------------------------------------
# OS Abstraction Layer: the ONE place that talks to the operating system.
#
# v1.0 shipped as pure Win32 (VirtualAlloc / CreateThread / CreateFileMapping).
# This module gives the identical primitives on Linux/glibc so the very same
# flat x86-64 machine code runs unmodified:
#
#   executable core   : mmap(PROT_READ|PROT_WRITE|PROT_EXEC)  vs VirtualAlloc
#   SMP spin workers  : pthread_create (libc)                 vs CreateThread
#   4.7..9.3 GB blob  : mmap(fd, PROT_READ)                   vs MapViewOfFile
#
# CALLING-CONVENTION BRIDGE (the real porting work)
# ------------------------------------------------
# Our assembly entry points were written for the Win64 ABI, i.e. the context
# pointer arrives in RCX. The Linux SysV ABI delivers the first integer argument
# in RDI. Rather than fork the assembly, we emit a 15-byte ABI trampoline per
# entry point at load time:
#
#     48 89 F9           mov  rcx, rdi      ; SysV arg1 -> our ABI
#     48 B8 <addr64>     movabs rax, target
#     FF E0              jmp  rax
#
# On Windows the same stub collapses to the 12-byte `movabs rax/jmp rax` form
# because arg1 is already RCX. This keeps a single canonical .asm source tree.
#
# Zero CRT, zero third-party deps: ctypes + libc only.
# ==============================================================================

import os
import sys
import mmap
import ctypes
import struct
import platform

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = not IS_WINDOWS

PAGE_EXECUTE_READWRITE = 0x40
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000

PROT_NONE = 0
PROT_READ = 1
PROT_WRITE = 2
PROT_EXEC = 4
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
MAP_FAILED = ctypes.c_void_p(-1).value

if IS_WINDOWS:
    _kernel32 = ctypes.windll.kernel32
    _kernel32.VirtualAlloc.restype = ctypes.c_void_p
    _kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_uint32, ctypes.c_uint32]
    _kernel32.VirtualFree.restype = ctypes.c_bool
    _kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    _kernel32.CreateThread.restype = ctypes.c_void_p
    _kernel32.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_bool
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _libc = None
else:
    _kernel32 = None
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32,
                           ctypes.c_int32, ctypes.c_int32, ctypes.c_long]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.sysconf.restype = ctypes.c_long
    # int pthread_create(pthread_t*, const pthread_attr_t*, void*(*)(void*), void*)
    _libc.pthread_create.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]
    _libc.pthread_join.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]

MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3

PAGE = 4096


# ------------------------------------------------------------------------------
# Executable memory
# ------------------------------------------------------------------------------
def exec_alloc(size):
    """RWX page-aligned region for flat machine code. Returns raw address."""
    size = (size + PAGE - 1) & ~(PAGE - 1)
    if IS_WINDOWS:
        addr = _kernel32.VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE,
                                      PAGE_EXECUTE_READWRITE)
    else:
        addr = _libc.mmap(None, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    if not addr or addr == MAP_FAILED:
        raise OSError(f"exec_alloc({size}) failed: {ctypes.get_errno() if IS_LINUX else 'win32'}")
    return addr


def exec_write(addr, code):
    """Copy machine code into an executable region (no mprotect needed: the
    region is created RWX, matching what v1.0 did with PAGE_EXECUTE_READWRITE)."""
    ctypes.memmove(addr, code, len(code))
    return addr


def exec_free(addr, size):
    size = (size + PAGE - 1) & ~(PAGE - 1)
    if IS_WINDOWS:
        _kernel32.VirtualFree(addr, 0, MEM_RELEASE)
    else:
        _libc.munmap(ctypes.c_void_p(addr), size)


# ------------------------------------------------------------------------------
# ABI trampoline: makes a SysV caller reach our Win64-style (RCX) entry points
# ------------------------------------------------------------------------------
def build_trampoline(target_addr, n_args=1):
    """Return (stub_addr, stub_len, CFUNCTYPE) for `target_addr(ctx_ptr)`."""
    code = b""
    if not IS_WINDOWS and n_args >= 1:
        code += b"\x48\x89\xF9"                     # mov rcx, rdi
    code += b"\x48\xB8" + struct.pack("<Q", target_addr)   # movabs rax, target
    code += b"\xFF\xE0"                                    # jmp rax
    code += b"\x90" * (16 - len(code))
    stub = exec_alloc(len(code) + PAGE)
    exec_write(stub, code)
    return stub, len(code), ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)


def make_entry(target_addr, restype=None, argtypes=(ctypes.c_void_p,)):
    """Wrap an asm entry point so Python calls it through the right ABI."""
    restype = restype if restype is not None else ctypes.c_uint64
    stub, _, _ = build_trampoline(target_addr, n_args=max(1, len(argtypes)))
    proto = ctypes.CFUNCTYPE(restype, *argtypes)
    fn = proto(stub)
    fn._stub = stub
    return fn


def make_reg_entry(target_addr, n_args, restype=ctypes.c_uint64):
    """Python-callable wrapper for a kernel taking 1..8 register arguments.

    The engine's internal convention (used by every inter-kernel CALL, so the
    engine itself never depends on the OS ABI) is Win64-style:
        PULSAR: rcx, rdx, r8, r9, [rsp+32], [rsp+40], [rsp+48]
    Linux delivers SysV:
        SysV  : rdi, rsi, rdx, rcx, r8, r9, then stack
    This emits the translating thunk; on Windows the two already agree and the
    thunk degenerates to a tail jump. Every emitted stub is disassembly-verified
    (see tests/test_bf16_gemb.py) because a wrong REX bit silently encodes a
    different register pair and shows up as garbage, not as an error.
    """
    if IS_WINDOWS:
        code = b"\x48\xB8" + struct.pack("<Q", target_addr) + b"\xFF\xE0"
    else:
        parts = []
        stack_in = n_args >= 7
        # 72 (not 80): keeps rsp 16-byte aligned at the CALL, as SysV requires.
        spill = 72 if stack_in else 56
        parts.append(b"\x48\x83\xEC\x38" if spill == 56 else
                     b"\x48\x81\xEC" + struct.pack("<I", spill))       # sub rsp, spill
        if stack_in:
            parts.append(b"\x48\x8B\x84\x24" + struct.pack("<I", spill + 8))
            parts.append(b"\x48\x89\x44\x24\x30")                    # mov [rsp+48], rax
        parts.append(b"\x4C\x89\x44\x24\x20")                       # mov [rsp+32], r8
        parts.append(b"\x4C\x89\x4C\x24\x28")                       # mov [rsp+40], r9
        if n_args < 7:
            # unstated trailing stack args must be defined (0), never garbage
            parts.append(b"\x48\xC7\x44\x24\x30" + struct.pack("<I", 0))
        parts += [b"\x49\x89\xD2",                                   # mov r10, rdx (arg3)
                  b"\x49\x89\xCB",                                   # mov r11, rcx (arg4)
                  b"\x48\x89\xF9",                                   # mov rcx, rdi
                  b"\x48\x89\xF2",                                   # mov rdx, rsi
                  b"\x4D\x89\xD0",                                   # mov r8,  r10
                  b"\x4D\x89\xD9"]                                   # mov r9,  r11
        parts.append(b"\x48\xB8" + struct.pack("<Q", target_addr))    # movabs rax, target
        # Always CALL (never a tail jmp): the frame was moved by `sub rsp`, so a
        # tail jump would make the kernel's RET pop our own frame instead of the
        # caller's return address.
        parts.append(b"\xFF\xD0")                                     # call rax
        parts.append(b"\x48\x83\xC4\x38" if spill == 56 else
                     b"\x48\x81\xC4" + struct.pack("<I", spill))       # add rsp, spill
        parts.append(b"\xC3")                                          # ret
        code = b"".join(parts)
    stub = exec_alloc(len(code) + 64)
    exec_write(stub, code)
    argtypes = [ctypes.c_void_p if i < 4 else ctypes.c_uint64 for i in range(n_args)]
    fn = ctypes.CFUNCTYPE(restype, *argtypes)(stub)
    fn._stub = stub
    fn._code = code
    return fn


_PTHREAD_STUBS = {}


def spawn_worker(entry_addr, arg_addr):
    """Start one spin-worker thread executing `entry_addr(ctx)`; return handle."""
    if IS_WINDOWS:
        h = _kernel32.CreateThread(None, 0, entry_addr, arg_addr, 0, None)
        if not h:
            raise OSError("CreateThread failed")
        return ("w", h, None)
    # SysV pthread start routine receives its argument in RDI -> trampoline fixes RCX
    key = entry_addr
    stub = _PTHREAD_STUBS.get(key)
    if stub is None:
        code = (b"\x48\x89\xF9" +                            # mov rcx, rdi
                b"\x48\xB8" + struct.pack("<Q", entry_addr) +  # movabs rax, target
                b"\xFF\xE0")                                 # jmp rax
        stub = exec_alloc(len(code) + PAGE)
        exec_write(stub, code)
        _PTHREAD_STUBS[key] = stub
    tid = ctypes.c_ulong(0)
    rc = _libc.pthread_create(ctypes.byref(tid), None, stub, ctypes.c_void_p(arg_addr))
    if rc != 0:
        raise OSError(f"pthread_create failed rc={rc}")
    return ("l", tid.value, None)


def join_worker(handle, timeout_ms=None):
    kind, h, _ = handle
    if kind == "w":
        _kernel32.WaitForSingleObject(h, 0xFFFFFFFF if timeout_ms is None else timeout_ms)
        _kernel32.CloseHandle(h)
    else:
        _libc.pthread_join(h, None)      # workers exit via stop_flag


# ------------------------------------------------------------------------------
# Read-only file mapping for the weight blob (page cache, zero copy)
# ------------------------------------------------------------------------------
class MappedFile:
    """mmap a weight blob; .addr is what the assembly dereferences."""

    def __init__(self, path, shared=False):
        self.path = path
        self.size = os.path.getsize(path)
        self._fd = os.open(path, os.O_RDONLY)
        flags = mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE
        self._mm = mmap.mmap(self._fd, self.size, flags=flags,
                             prot=mmap.PROT_READ)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        if not IS_LINUX and self.size:
            # Python's mmap on Windows cannot expose an address; fall back to mapping
            # the file through the Win32 file-mapping API.
            self._mm.close(); os.close(self._fd)
            self.addr = _win_map(path)

    def touch(self, step=1 << 20):
        """Fault every page in so first-token latency is not charged to page faults."""
        buf = (ctypes.c_char * 1)
        for off in range(0, self.size, step):
            ctypes.cast(self.addr + off, ctypes.POINTER(buf))[0]
        return self.size

    def advise_sequential(self, start=0, length=None):
        if IS_LINUX:
            _libc.madvise(ctypes.c_void_p(self.addr + start),
                          length or (self.size - start), MADV_SEQUENTIAL)

    def close(self):
        try:
            self._mm.close()
            os.close(self._fd)
        except Exception:
            pass


def _win_map(path):
    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    PAGE_READONLY = 2
    FILE_MAP_READ = 4
    h = _kernel32.CreateFileW(path, GENERIC_READ, 1, None, OPEN_EXISTING, 128, None)
    m = _kernel32.CreateFileMappingW(h, None, PAGE_READONLY, 0, 0, None)
    return _kernel32.MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0)


# ------------------------------------------------------------------------------
# Environment report (replaces the Win32-only CPUID print in v1.0)
# ==============================================================================
def cpu_report():
    if IS_LINUX:
        feats = set()
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("flags") or line.startswith("Features"):
                    feats = set(line.split(":", 1)[1].split())
                    break
        except OSError:
            pass
        model = "unknown x86-64"
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        online = os.cpu_count() or 1
        try:
            online = sum(1 for ln in open("/proc/cpuinfo")
                         if ln.startswith("processor")) or online
        except OSError:
            pass
        return {"model": model, "avx2": "avx2" in feats, "f16c": "f16c" in feats,
                "fma": "fma" in feats, "threads": online, "os": f"Linux {platform.release()}"}
    return {"model": platform.processor() or "unknown", "avx2": True, "f16c": True,
            "fma": True, "threads": os.cpu_count() or 1, "os": "Windows"}


def require_avx2_fma():
    r = cpu_report()
    if not (r["avx2"] and r["fma"]):
        raise RuntimeError(f"AVX2+FMA3 required, got: {r}")
    return r


# ------------------------------------------------------------------------------
# Module loading: assemble (optional) + map + resolve the export trailer
# ------------------------------------------------------------------------------
class Module:
    def __init__(self, name, path, addr, size, exports):
        self.name = name
        self.path = path
        self.addr = addr
        self.size = size
        self.exports = exports           # {name: absolute address}

    def off(self, name):
        return self.exports[name] - self.addr

    def entry(self, name, restype=ctypes.c_uint64, argtypes=(ctypes.c_void_p,)):
        """Callable asm entry point, ABI-correct for the running OS."""
        return make_entry(self.exports[name], restype=restype, argtypes=argtypes)

    def __repr__(self):
        return f"<PULSAR module {self.name} {self.size:,}B at 0x{self.addr:X} " \
               f"{len(self.exports)} exports>"


def parse_trailer(image):
    """PLSE trailer: [ ...code... ][ dd off*N ][ dd nameoff*N ][ dd N ][ 'PLSE' ]."""
    if len(image) < 12 or image[-4:] != b"PLSE":
        return None
    n = struct.unpack_from("<I", image, len(image) - 8)[0]
    if n == 0 or n > 4096 or len(image) < 8 + 8 * n:
        return None
    offs_at = len(image) - 8 - 8 * n
    names_at = offs_at + 4 * n
    exports = {}
    for i in range(n):
        o = struct.unpack_from("<I", image, offs_at + 4 * i)[0]
        p = struct.unpack_from("<I", image, names_at + 4 * i)[0]
        end = image.index(b"\x00", p)
        exports[image[p:end].decode()] = o
    return exports


def assemble(asm_path, bin_path=None, fasm="fasm"):
    """Run FASM on a *_flat.asm module. Returns the .bin path used."""
    import subprocess
    if bin_path is None:
        bin_path = os.path.splitext(asm_path)[0].replace("_flat", "") + ".bin"
    os.makedirs(os.path.dirname(os.path.abspath(bin_path)) or ".", exist_ok=True)
    r = subprocess.run([fasm, "-m", "65536", asm_path, bin_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fasm failed on {asm_path}:\n{r.stdout}\n{r.stderr}")
    return bin_path


def load_module(path, name=None, asm_path=None):
    """Load a PULSAR module. `path` may be a .bin, or a .asm (assembled on demand)."""
    if path.endswith(".asm"):
        path = assemble(path)
    with open(path, "rb") as f:
        image = f.read()
    exports = parse_trailer(image)
    code_len = len(image) - 8 - 8 * struct.unpack_from("<I", image, len(image) - 8)[0] \
        if exports else len(image)
    addr = exec_alloc(len(image) + 64)
    exec_write(addr, image)
    abs_exports = {k: addr + v for k, v in (exports or {}).items()}
    return Module(name or os.path.basename(path), path, addr, code_len, abs_exports)
