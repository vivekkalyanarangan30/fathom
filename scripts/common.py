"""Shared helpers: model loading, calibration text, per-head Q/K/V capture."""
import os, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(name="Qwen/Qwen3-1.7B", dtype=None, attn="eager"):
    """attn='sdpa' for long-context captures (eager materialises T x T scores); the patched harness replaces attention anyway."""
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"device_map": "cuda"} if torch.cuda.is_available() else {}   # L4 VM has 15 GB RAM: stream weights straight to the GPU
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn, **kw)
    model.eval()
    if not torch.cuda.is_available(): model.to(device())
    return model, tok

def wikitext(split="test", n_chars=None):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return text if n_chars is None else text[:n_chars]

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
def render_msg(m):
    """One OpenHands trajectory message -> ChatML text (Qwen2.5 convention; tool results as user-role tool_response blocks, tool calls inlined)."""
    role = m.get("role", "user"); content = m.get("content") or ""
    if isinstance(content, list): content = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if role == "tool": return f"{IM_S}user\n<tool_response>\n{content}\n</tool_response>{IM_E}\n"
    if role == "assistant":
        for c in m.get("tool_calls") or []:
            f = c.get("function", c) if isinstance(c, dict) else {}
            name = f.get("name", ""); args = f.get("arguments", "")
            content += f"\n<tool_call>\n{json.dumps({'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args)})}\n</tool_call>"
        return f"{IM_S}assistant\n{content}{IM_E}\n"
    return f"{IM_S}{role}\n{content}{IM_E}\n"

def rope(q, cos, sin):
    x1, x2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return q * cos + rot * sin

@torch.no_grad()
def capture_qkv(model, input_ids):
    """Return per-layer post-RoPE q,k (after q_norm/k_norm) and v for one sequence: lists of [H,T,D]/[Hkv,T,D]."""
    cfg = model.config
    caps = {}
    hooks = []
    def mk(i):
        def hook(mod, args, kwargs, out):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs["position_embeddings"]
            B, T, _ = hs.shape
            D = getattr(mod, "head_dim", None) or model.config.head_dim
            qn = getattr(mod, "q_norm", None) or (lambda x: x); kn = getattr(mod, "k_norm", None) or (lambda x: x)   # Llama/Mistral have no QK-norm
            q = qn(mod.q_proj(hs).view(B, T, -1, D)).transpose(1, 2)
            k = kn(mod.k_proj(hs).view(B, T, -1, D)).transpose(1, 2)
            v = mod.v_proj(hs).view(B, T, -1, D).transpose(1, 2)
            cos, sin = pe[0].unsqueeze(1), pe[1].unsqueeze(1)
            caps[i] = (rope(q, cos, sin)[0].float().cpu(), rope(k, cos, sin)[0].float().cpu(), v[0].float().cpu(),
                       k[0].float().cpu())
        return hook
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(mk(i), with_kwargs=True))
    model(input_ids.to(device()))
    for h in hooks: h.remove()
    return [caps[i] for i in range(len(model.model.layers))]
