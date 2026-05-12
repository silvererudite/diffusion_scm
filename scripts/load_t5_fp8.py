"""
load_t5_fp8.py
--------------
Load T5-XXL from the SD3 fp8 safetensors file. Upcast to bf16 for inference
since PyTorch's fp8 op support is still partial.

Disk:  ~5 GB download
GPU:   ~20 GB in bf16 at runtime (fits comfortably on A10G 24GB, A100, etc.)

Replace the T5 loading block in 03c_multi_encoder_axes.py with this.
"""
import torch
from transformers import T5EncoderModel, T5TokenizerFast, T5Config
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def load_t5_xxl_fp8(device="cuda"):
    print("Downloading T5-XXL fp8 weights from stabilityai/stable-diffusion-3-medium...")
    t5_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-3-medium",
        filename="text_encoders/t5xxl_fp8_e4m3fn.safetensors",
    )
    print(f"  -> {t5_path}")

    print("Loading tokenizer (small, ~5MB)...")
    tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    print("Building empty T5 encoder shell and upcasting fp8 weights to bf16...")
    cfg = T5Config.from_pretrained("google/t5-v1_1-xxl")
    # Construct empty model in bf16 — we'll fill weights from the fp8 file
    model = T5EncoderModel(cfg).to(torch.bfloat16)

    # Load the fp8 safetensors. Each tensor will be torch.float8_e4m3fn.
    state = load_file(t5_path)

    # Drop decoder/lm_head keys — the SD3 file is encoder-only but be defensive
    state = {k: v for k, v in state.items()
             if not k.startswith("decoder.") and not k.startswith("lm_head.")}

    # Upcast every fp8 tensor to bf16 before loading
    upcast = {}
    for k, v in state.items():
        if v.dtype == torch.float8_e4m3fn:
            # fp8 tensors can't be operated on directly with most ops.
            # Convert to fp32 first (safe), then to bf16.
            upcast[k] = v.to(torch.float32).to(torch.bfloat16)
        else:
            upcast[k] = v.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(upcast, strict=False)
    print(f"  Loaded. Missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing[:3]:
        print(f"  Missing samples: {missing[:3]}")
    if unexpected[:3]:
        print(f"  Unexpected samples: {unexpected[:3]}")

    model = model.to(device).eval()
    return model, tok


def t5_encode_factory(model, tokenizer, device="cuda"):
    """Return an encode_fn(texts) -> (B, 4096) float32 numpy array."""
    import numpy as np
    def encode(texts):
        with torch.no_grad():
            tok = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(device)
            out = model(**tok).last_hidden_state          # (B, T, 4096) bf16
            mask = tok.attention_mask.unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.cpu().float().numpy().astype(np.float32)
    return encode


if __name__ == "__main__":
    # Quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_t5_xxl_fp8(device)
    encode = t5_encode_factory(model, tok, device)
    out = encode(["a photo of a kind person", "a photo of a competent person"])
    print(f"Output shape: {out.shape}  dtype: {out.dtype}")
    print(f"Norms: {[float((out[i]**2).sum()**0.5) for i in range(len(out))]}")
    # Cosine between the two should be high but not 1 — they're related concepts
    a, b = out[0], out[1]
    cos = (a @ b) / ((a @ a)**0.5 * (b @ b)**0.5)
    print(f"Cosine(kind, competent): {cos:.3f}")