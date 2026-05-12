"""
diagnose_clip.py
----------------
Tiny script to see what CLIP.get_text_features actually returns on this env.
"""
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP-L...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

with torch.no_grad():
    tok = proc(text=["a photo of a warm person"], return_tensors="pt", padding=True).to(device)
    out = clip.get_text_features(**tok)

print(f"Type:  {type(out)}")
print(f"Is tensor: {isinstance(out, torch.Tensor)}")

# If it's a wrapper, inspect attributes
if not isinstance(out, torch.Tensor):
    print("Attributes:")
    for attr in dir(out):
        if attr.startswith("_"): continue
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            print(f"  .{attr}  -> Tensor of shape {tuple(val.shape)}")
        elif val is None:
            print(f"  .{attr}  -> None")
        else:
            print(f"  .{attr}  -> {type(val).__name__}")

import transformers
print(f"\ntransformers version: {transformers.__version__}")