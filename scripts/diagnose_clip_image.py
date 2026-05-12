"""
diagnose_clip_image.py
----------------------
Verify whether get_image_features on transformers 5.x has the same wrapper
issue. If yes, your existing CLIP image embeddings in embeddings.npz are
the pre-projection vectors, not the aligned shared-space ones, and need to
be re-computed.
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP-L...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Make a dummy image
img = Image.new("RGB", (224, 224), color="white")

with torch.no_grad():
    inputs = proc(images=[img], return_tensors="pt").to(device)
    out = clip.get_image_features(**inputs)

print(f"Type: {type(out)}")
print(f"Is tensor: {isinstance(out, torch.Tensor)}")

if not isinstance(out, torch.Tensor):
    print("Attributes (non-None tensors only):")
    for attr in dir(out):
        if attr.startswith("_"): continue
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            print(f"  .{attr}  -> Tensor of shape {tuple(val.shape)}")

# Verify what your saved embeddings look like
print("\n=== Existing embeddings on disk ===")
e = np.load("outputs/embeddings.npz", allow_pickle=True)
clip_emb = e["clip_emb"]
print(f"clip_emb shape: {clip_emb.shape}")
print(f"clip_emb dtype: {clip_emb.dtype}")
print(f"clip_emb norms (first 5): {np.linalg.norm(clip_emb[:5], axis=-1)}")
# If they were properly projected, norms should be ~1.0 (we L2-normalized).
# Dimensionality: projected = 768, but the projection input is also 1024 for ViT-L.
# Actually for CLIP-L: vision tower output is 1024, projection is 1024->768.
# So if shape is (N, 768) the projection MAY have been applied.
# If shape is (N, 1024) then definitely raw pooler output.

# Compare to what we'd get with the correct (projected) path
with torch.no_grad():
    inputs = proc(images=[img], return_tensors="pt").to(device)
    out = clip.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        feats_proj = out
    else:
        feats_proj = clip.visual_projection(out.pooler_output)
    feats_proj_normed = feats_proj / feats_proj.norm(dim=-1, keepdim=True)
print(f"\nManually projected dummy-image embedding shape: {feats_proj.shape}")
print(f"Manually projected dim: {feats_proj.shape[-1]}")
print(f"  -> if this matches clip_emb.shape[1]={clip_emb.shape[1]}, projection MAY have been applied;")
print(f"     but you should still verify the *values* match by running one image through both paths.")