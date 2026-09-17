"""Lazy safetensors reader without mmap (safe_open mmaps the whole file, which the kernel refuses for a 17 GB file on a 15 GB VM):
reads only the requested leading-dim range with one pread. Drop-in for the subset of the safe_open API these scripts use."""
import json, struct, os, numpy as np, torch
DT = {"F16": np.float16, "F32": np.float32, "BF16": None}
class _Slice:
    def __init__(self, f, meta): self.f, self.meta = f, meta
    def get_shape(self): return list(self.meta["shape"])
    def __getitem__(self, idx):
        shape = self.meta["shape"]; idx = idx if isinstance(idx, tuple) else (idx,)
        lead = idx[0]; rest = idx[1:]
        n0 = shape[0]
        if isinstance(lead, int): r0, r1, squeeze = lead % n0, lead % n0 + 1, True
        else: r0, r1, _ = lead.indices(n0); squeeze = False
        row = int(np.prod(shape[1:])) * 2; start = self.meta["data_offsets"][0] + r0 * row
        buf = os.pread(self.f.fileno(), (r1 - r0) * row, self.f.base + start)
        t = torch.from_numpy(np.frombuffer(buf, dtype=DT[self.meta["dtype"]]).reshape(r1 - r0, *shape[1:]).copy())
        if squeeze: return t[0][rest] if rest else t[0]
        return t[(slice(None),) + rest] if rest else t
class CapsFile:
    def __init__(self, path):
        self.f = open(path, "rb"); n = struct.unpack("<Q", self.f.read(8))[0]; self.hdr = json.loads(self.f.read(n)); self.hdr.pop("__metadata__", None); self.f.base = 8 + n
    def keys(self): return list(self.hdr)
    def get_slice(self, name): return _Slice(self.f, self.hdr[name])
    def get_tensor(self, name): return self.get_slice(name)[:]
    def __getitem__(self, name): return self.get_tensor(name)
    def __contains__(self, name): return name in self.hdr
    def __iter__(self): return iter(self.hdr)
    def __len__(self): return len(self.hdr)
def safe_open(path, framework="pt", device="cpu"): return CapsFile(path)
def load_file(path): return CapsFile(path)      # lazy stand-in: per-tensor loads on access
