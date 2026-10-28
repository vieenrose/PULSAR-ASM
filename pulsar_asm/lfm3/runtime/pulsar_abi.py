# ==============================================================================
# Project PULSAR-ASM | runtime/pulsar_abi.py
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
MAP_SHARED = 0x01

MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3

PAGE = 4096

if IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _kernel32.VirtualAlloc.restype = ctypes.c_void_p
    _kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    _kernel32.VirtualFree.restype = ctypes.c_bool
    _kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CreateFileMappingW.restype = ctypes.c_void_p
    _kernel32.MapViewOfFile.restype = ctypes.c_void_p
    _kernel32.UnmapViewOfFile.restype = ctypes.c_bool
    _kernel32.CloseHandle.restype = ctypes.c_bool
    _kernel32.CreateThread.restype = ctypes.c_void_p
    _kernel32.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_bool
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _libc = None
else:
    _kernel32 = None
    _libc = ctypes.CDLL("libpthread.so.0", use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32,
                           ctypes.c_int32, ctypes.c_int32, ctypes.c_long]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.sysconf.restype = ctypes.c_long
    _libc.pthread_create.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]
    _libc.pthread_join.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]


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


class Module:
    def __init__(self, name, path, addr, size, exports):
        self.name = name
        self.path = path
        self.addr = addr
        self.size = size
        self.exports = exports

    def off(self, name):
        return self.exports[name] - self.addr

    def entry(self, name, restype=ctypes.c_uint64, argtypes=(ctypes.c_void_p,)):
        return make_entry(self.exports[name], restype=restype, argtypes=argtypes)

    def __repr__(self):
        return f"<PULSAR module {self.name} {self.size:,}B at 0x{self.addr:X} " \
               f"{len(self.exports)} exports>"


class MappedFile:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        if IS_LINUX:
            fd = os.open(path, os.O_RDONLY)
            try:
                self.addr = _libc.mmap(0, self.size, PROT_READ, MAP_SHARED, fd, 0)
            finally:
                os.close(fd)
            if self.addr == -1:
                raise OSError("mmap failed")
        else:
            self.addr = _win_map(path)
        self._buf = (ctypes.c_ubyte * self.size).from_address(self.addr)

    def __len__(self):
        return self.size

    def close(self):
        if IS_LINUX:
            _libc.munmap(ctypes.c_void_p(self.addr), self.size)
        else:
            _kernel32.UnmapViewOfFile(ctypes.c_void_p(self.addr))
        self.addr = 0


def exec_alloc(size):
    size = (size + PAGE - 1) & ~(PAGE - 1)
    if IS_LINUX:
        addr = _libc.mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        if addr == -1 or addr is None:
            raise OSError(f"mmap exec failed, size={size}")
        return addr
    else:
        addr = _kernel32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
        if not addr:
            raise OSError("VirtualAlloc failed")
        return addr


def exec_write(addr, data):
    mv = (ctypes.c_ubyte * len(data)).from_address(addr)
    mv[:] = bytes(data)


def _win_map(path):
    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    PAGE_READONLY = 2
    FILE_MAP_READ = 4
    h = _kernel32.CreateFileW(path, GENERIC_READ, 1, None, OPEN_EXISTING, 128, None)
    m = _kernel32.CreateFileMappingW(h, None, PAGE_READONLY, 0, 0, None)
    return _kernel32.MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0)


def make_entry(addr, restype=ctypes.c_uint64, argtypes=(ctypes.c_void_p,)):
    class _Call(ctypes.Structure):
        _fields_ = [("jmp", ctypes.c_ubyte), ("rel", ctypes.c_int32)]
    if IS_WINDOWS:
        stub = (ctypes.c_ubyte * 12)()
        stub[0] = 0x48; stub[1] = 0xB8
        ctypes.memmove(ctypes.addressof((ctypes.c_ubyte*8).from_buffer(stub[2:10])), ctypes.addressof(ctypes.c_uint64(addr)), 8)
        stub[10] = 0xFF; stub[11] = 0xE0
    else:
        stub = (ctypes.c_ubyte * 15)()
        stub[0] = 0x48; stub[1] = 0x89; stub[2] = 0xF9
        stub[3] = 0x48; stub[4] = 0xB8
        ctypes.memmove(ctypes.addressof((ctypes.c_ubyte*8).from_buffer(stub[5:13])), ctypes.addressof(ctypes.c_uint64(addr)), 8)
        stub[13] = 0xFF; stub[14] = 0xE0
    stub_addr = exec_alloc(len(stub))
    exec_write(stub_addr, bytes(stub))
    ftype = ctypes.CFUNCTYPE(restype, *argtypes)
    return ftype(stub_addr)


def make_reg_entry(addr, nargs):
    argtypes = (ctypes.c_void_p,) * nargs
    return make_entry(addr, restype=ctypes.c_uint64, argtypes=argtypes)


def parse_trailer(image):
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
        try:
            end = image.index(b"\x00", p)
        except ValueError:
            continue
        exports[image[p:end].decode()] = o
    return exports


def assemble(asm_path, bin_path=None, fasm="fasm"):
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
    if path.endswith(".asm"):
        path = assemble(path)
    with open(path, "rb") as f:
        image = f.read()
    exports = parse_trailer(image)
    if not exports:
        raise RuntimeError(f"no PLSE trailer in {path}")
    n = struct.unpack_from("<I", image, len(image) - 8)[0]
    code_len = len(image) - 8 - 8 * n
    addr = exec_alloc(len(image) + 64)
    exec_write(addr, image)
    abs_exports = {k: addr + v for k, v in (exports or {}).items()}
    return Module(name or os.path.basename(path), path, addr, code_len, abs_exports)


def spawn_worker(func, arg_addr):
    if IS_LINUX:
        stub = (ctypes.c_ubyte * 32)()
        stub[0] = 0x48; stub[1] = 0xB8
        ctypes.memmove(ctypes.addressof((ctypes.c_ubyte*8).from_buffer(stub[2:10])), ctypes.addressof(ctypes.c_uint64(func)), 8)
        stub[10] = 0x48; stub[11] = 0xBF
        ctypes.memmove(ctypes.addressof((ctypes.c_ubyte*8).from_buffer(stub[12:20])), ctypes.addressof(ctypes.c_uint64(arg_addr)), 8)
        stub[20] = 0xFF; stub[21] = 0xE0
        stub[22] = 0x31; stub[23] = 0xC0
        stub[24] = 0xC3
        stub_addr = exec_alloc(64)
        exec_write(stub_addr, bytes(stub))
        tid = ctypes.c_ulong()
        rc = _libc.pthread_create(ctypes.byref(tid), None, stub_addr, None)
        if rc != 0:
            raise OSError(f"pthread_create failed {rc}")
        return int(tid)
    else:
        raise NotImplementedError("worker threads only on Linux here")


def join_worker(h):
    if IS_LINUX:
        _libc.pthread_join(h, None)
    else:
        _kernel32.WaitForSingleObject(h, 0xFFFFFFFF)
        _kernel32.CloseHandle(h)
