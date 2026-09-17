"""L4 version of 00_capture.py: identical tensors (post-RoPE q/k after q_norm/k_norm, v, pre-RoPE k; fp16; names q{L},k{L},v{L},kpre{L})
but streamed layer-by-layer into a safetensors file so peak CPU RAM is one layer (the VM has 15 GB; 8B@32k is 34 GB of captures).
Usage: python scripts/l4_capture.py <hf-model> <T> [train,test]"""
import os, sys, json, struct, math, torch
from common import *
torch.set_grad_enabled(False)

if torch.cuda.is_available():                                                           # a materialised 128k x 128k mask OOMs; one unpadded sequence -> is_causal is exact
    import transformers.integrations.sdpa_attention as _sa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
    _orig_sdpa = _sa.sdpa_attention_forward
    def _sdpa_causal(module, query, key, value, attention_mask, **kw):
        kw["is_causal"] = query.shape[2] > 1 if kw.get("is_causal") is None else kw["is_causal"]
        return _orig_sdpa(module, query, key, value, None, **kw)
    _AAF["sdpa"] = _sdpa_causal; torch.backends.cuda.enable_math_sdp(False)
TEXT_DIR = SUFFIX = None                                                                 # --text-dir DIR --suffix S: capture {split}.txt instead of Wikitext, as caps_<model>-S_...
for flag in ("--text-dir", "--suffix"):
    if flag in sys.argv:
        i = sys.argv.index(flag); val = sys.argv[i + 1]; del sys.argv[i:i + 2]
        if flag == "--text-dir": TEXT_DIR = val
        else: assert val.isalnum(), "--suffix must be alphanumeric"; SUFFIX = val
assert (TEXT_DIR is None) == (SUFFIX is None), "--text-dir and --suffix go together"
name = sys.argv[1]; T = int(sys.argv[2]); splits = (sys.argv[3] if len(sys.argv) > 3 else "train,test").split(",")
model, tok = load_model(name, attn="sdpa"); cfg = model.config
nL = cfg.num_hidden_layers; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads; D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
print(f"{name}: layers={nL} Hq={Hq} Hkv={Hkv} D={D} G={Hq//Hkv} qk_norm={hasattr(model.model.layers[0].self_attn, 'q_norm')}", flush=True)

class StreamWriter:
    """Writes a safetensors file tensor-by-tensor; header (dtype/shape/offsets) is fixed up front from the specs."""
    def __init__(self, path, specs):
        self.hdr, off = {}, 0
        for nm, shape in specs:
            nb = 2 * math.prod(shape); self.hdr[nm] = {"dtype": "F16", "shape": list(shape), "data_offsets": [off, off + nb]}; off += nb
        hb = json.dumps(self.hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        self.f = open(path, "wb"); self.f.write(struct.pack("<Q", len(hb))); self.f.write(hb); self.base = 8 + len(hb); self.total = off
    def write(self, nm, t, row0=0):
        """Write tensor t (fp16, cpu) as rows [row0, row0+len(t)) of the leading dim of tensor nm."""
        shp = self.hdr[nm]["shape"]; assert list(t.shape[1:]) == shp[1:] and t.dtype == torch.float16 and row0 + t.shape[0] <= shp[0], (nm, t.shape, row0)
        self.f.seek(self.base + self.hdr[nm]["data_offsets"][0] + row0 * 2 * math.prod(shp[1:])); self.f.write(memoryview(t.contiguous().numpy()))
    def close(self): self.f.flush(); self.f.close()

def capture_stream(ids, path):
    specs = []
    for L in range(nL): specs += [(f"q{L}", (Hq, T, D)), (f"k{L}", (Hkv, T, D)), (f"v{L}", (Hkv, T, D)), (f"kpre{L}", (Hkv, T, D))]
    W = StreamWriter(path, specs); hooks = []
    def mk(i):
        def hook(mod, args, kwargs, out):
            hs = kwargs.get("hidden_states", args[0] if args else None); pe = kwargs["position_embeddings"]
            B, Tt, _ = hs.shape
            qn = getattr(mod, "q_norm", None) or (lambda x: x); kn = getattr(mod, "k_norm", None) or (lambda x: x)
            cos, sin = pe[0].unsqueeze(1).float(), pe[1].unsqueeze(1).float()
            q = qn(mod.q_proj(hs).view(B, Tt, -1, D)).transpose(1, 2)[0]           # [Hq, T, D] model dtype; RoPE in fp32 per chunk of 8 heads (memory)
            for h0 in range(0, Hq, 8): W.write(f"q{i}", rope(q[h0:h0 + 8].float(), cos[0], sin[0]).half().cpu(), row0=h0)
            del q
            k = kn(mod.k_proj(hs).view(B, Tt, -1, D)).transpose(1, 2)[0].float()
            W.write(f"kpre{i}", k.half().cpu()); W.write(f"k{i}", rope(k, cos[0], sin[0]).half().cpu()); del k
            W.write(f"v{i}", mod.v_proj(hs).view(B, Tt, -1, D).transpose(1, 2)[0].half().cpu())
        return hook
    for i, layer in enumerate(model.model.layers): hooks.append(layer.self_attn.register_forward_hook(mk(i), with_kwargs=True))
    model(ids.to(device()), use_cache=False, logits_to_keep=1)
    for h in hooks: h.remove()
    W.close(); print(f"wrote {path} ({W.total/1e9:.2f} GB)", flush=True)

for split in splits:
    off = 0 if split == "train" else 1000
    text = wikitext(split, 600_000 if T <= 16384 else 1_200_000) if TEXT_DIR is None else open(os.path.join(TEXT_DIR, f"{split}.txt"), encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[:, off:off + T]
    assert ids.shape[1] == T, ids.shape
    base = name.split('/')[-1] + (f"-{SUFFIX}" if SUFFIX else "")
    t0 = time.time(); capture_stream(ids, f"{RES}/caps_{base}_{split}_T{T}.safetensors")
    print(f"saved {split} T={T} in {time.time()-t0:.0f}s; peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    torch.cuda.empty_cache()
