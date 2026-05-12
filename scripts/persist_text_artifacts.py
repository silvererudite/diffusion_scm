"""
03d_persist_text_artifacts.py
-----------------------------
Re-runs bigG and T5 just long enough to SAVE their axis vectors and the
group-name embeddings to disk, so downstream image-side and PGM scripts
never have to reload these encoders.

What this saves to text_artifacts.npz:
  clipL_v_warmth, clipL_v_competence       (768,)
  bigG_v_warmth,  bigG_v_competence        (1280,)
  t5_v_warmth,    t5_v_competence          (4096,)
  bigG_group_emb, t5_group_emb             (n_groups, D)  group-name embeddings
  clipL_group_emb                          (n_groups, 768)
  group_keys                               (n_groups,)    aligned to all above

Also writes text_artifacts_summary.json with axis angles and correlations,
so you have a single source of truth for the text-side results.

Run with --include_t5 to also persist T5; otherwise just CLIP-L + bigG.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

WARMTH_POS = ["warm","friendly","sincere","trustworthy","kind","caring",
              "honest","good-natured","tolerant","sociable","generous",
              "compassionate","approachable"]
WARMTH_NEG = ["cold","unfriendly","insincere","untrustworthy","cruel",
              "callous","dishonest","mean","intolerant","antisocial",
              "selfish","hostile","unapproachable"]
COMPETENCE_POS = ["competent","skilled","intelligent","capable","efficient",
                  "confident","knowledgeable","clever","independent",
                  "ambitious","successful","professional","accomplished"]
COMPETENCE_NEG = ["incompetent","unskilled","unintelligent","incapable",
                  "inefficient","insecure","ignorant","foolish","dependent",
                  "unambitious","unsuccessful","unprofessional","failed"]

PROMPT_TEMPLATE = "a photo of a {} person"
GROUP_TEMPLATE  = "a photo of a {}"

def bipolar_axis(encode_fn, pos, neg):
    p = encode_fn([PROMPT_TEMPLATE.format(w) for w in pos])
    n = encode_fn([PROMPT_TEMPLATE.format(w) for w in neg])
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    v = p.mean(0) - n.mean(0)
    return v / np.linalg.norm(v)

def correlate(coords, gt):
    w_pred, c_pred = coords[:, 0], coords[:, 1]
    w_r, w_p = pearsonr(w_pred, gt["warmth_z"])
    c_r, c_p = pearsonr(c_pred, gt["competence_z"])
    return {"r_w": float(w_r), "p_w": float(w_p),
            "r_c": float(c_r), "p_c": float(c_p)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_t5", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gt = pd.read_csv("outputs/groundtruth_clean.csv")
    emb = np.load("outputs/embeddings.npz", allow_pickle=True)
    img_cats = set(np.unique(emb["categories"]).tolist())
    gt = gt[gt["category_key"].isin(img_cats)].reset_index(drop=True)
    group_keys = gt["category_key"].tolist()
    group_prompts = [GROUP_TEMPLATE.format(k) for k in group_keys]
    print(f"Persisting text artifacts for {len(group_keys)} groups")

    artifacts = {"group_keys": np.array(group_keys)}
    summary = {"n_groups": len(group_keys)}

    # ========== CLIP-L ==========
    print("\nCLIP-L...")
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def clip_text(texts):
        with torch.no_grad():
            tok = cproc(text=texts, return_tensors="pt", padding=True).to(args.device)
            out = clip.get_text_features(**tok)
            if isinstance(out, torch.Tensor):
                feats = out
            else:
                # transformers 5.x wraps the output. pooler_output may be raw
                # (needs projection) or already projected — check the dim.
                proj_in_dim = clip.text_projection.weight.shape[1]
                proj_out_dim = clip.text_projection.weight.shape[0]
                p = out.pooler_output
                if p.shape[-1] == proj_in_dim:
                    feats = clip.text_projection(p)
                elif p.shape[-1] == proj_out_dim:
                    feats = p
                else:
                    raise RuntimeError(f"pooler_output dim {p.shape[-1]} matches neither "
                                       f"projection input ({proj_in_dim}) nor output ({proj_out_dim})")
        return feats.cpu().numpy().astype(np.float32)
    artifacts["clipL_v_warmth"]     = bipolar_axis(clip_text, WARMTH_POS, WARMTH_NEG)
    artifacts["clipL_v_competence"] = bipolar_axis(clip_text, COMPETENCE_POS, COMPETENCE_NEG)
    grp = clip_text(group_prompts)
    grp = grp / np.linalg.norm(grp, axis=-1, keepdims=True)
    artifacts["clipL_group_emb"] = grp
    coords = np.stack([grp @ artifacts["clipL_v_warmth"],
                       grp @ artifacts["clipL_v_competence"]], axis=1)
    summary["clipL"] = correlate(coords, gt)
    summary["clipL"]["wc_angle_deg"] = float(np.degrees(np.arccos(np.clip(
        artifacts["clipL_v_warmth"] @ artifacts["clipL_v_competence"], -1, 1))))
    print(f"  CLIP-L  angle={summary['clipL']['wc_angle_deg']:.1f}°  "
          f"r_W={summary['clipL']['r_w']:+.3f}  r_C={summary['clipL']['r_c']:+.3f}")
    del clip, cproc; torch.cuda.empty_cache()

    # ========== bigG ==========
    print("\nOpenCLIP bigG...")
    import open_clip
    bigg, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=args.device)
    bigg.eval()
    btok = open_clip.get_tokenizer("ViT-bigG-14")
    def bigg_text(texts):
        with torch.no_grad():
            tok = btok(texts).to(args.device)
            feats = bigg.encode_text(tok)
        return feats.cpu().numpy().astype(np.float32)
    artifacts["bigG_v_warmth"]     = bipolar_axis(bigg_text, WARMTH_POS, WARMTH_NEG)
    artifacts["bigG_v_competence"] = bipolar_axis(bigg_text, COMPETENCE_POS, COMPETENCE_NEG)
    grp = bigg_text(group_prompts)
    grp = grp / np.linalg.norm(grp, axis=-1, keepdims=True)
    artifacts["bigG_group_emb"] = grp
    coords = np.stack([grp @ artifacts["bigG_v_warmth"],
                       grp @ artifacts["bigG_v_competence"]], axis=1)
    summary["bigG"] = correlate(coords, gt)
    summary["bigG"]["wc_angle_deg"] = float(np.degrees(np.arccos(np.clip(
        artifacts["bigG_v_warmth"] @ artifacts["bigG_v_competence"], -1, 1))))
    print(f"  bigG    angle={summary['bigG']['wc_angle_deg']:.1f}°  "
          f"r_W={summary['bigG']['r_w']:+.3f}  r_C={summary['bigG']['r_c']:+.3f}")
    del bigg, btok; torch.cuda.empty_cache()

    # ========== T5-XXL (optional) ==========
    if args.include_t5:
        print("\nT5-XXL fp8...")
        from load_t5_fp8 import load_t5_xxl_fp8, t5_encode_factory
        t5_model, t5_tok = load_t5_xxl_fp8(args.device)
        t5_encode = t5_encode_factory(t5_model, t5_tok, args.device)
        artifacts["t5_v_warmth"]     = bipolar_axis(t5_encode, WARMTH_POS, WARMTH_NEG)
        artifacts["t5_v_competence"] = bipolar_axis(t5_encode, COMPETENCE_POS, COMPETENCE_NEG)
        grp = t5_encode(group_prompts)
        grp = grp / np.linalg.norm(grp, axis=-1, keepdims=True)
        artifacts["t5_group_emb"] = grp
        coords = np.stack([grp @ artifacts["t5_v_warmth"],
                           grp @ artifacts["t5_v_competence"]], axis=1)
        summary["t5"] = correlate(coords, gt)
        summary["t5"]["wc_angle_deg"] = float(np.degrees(np.arccos(np.clip(
            artifacts["t5_v_warmth"] @ artifacts["t5_v_competence"], -1, 1))))
        print(f"  T5-XXL  angle={summary['t5']['wc_angle_deg']:.1f}°  "
              f"r_W={summary['t5']['r_w']:+.3f}  r_C={summary['t5']['r_c']:+.3f}")
        del t5_model, t5_tok; torch.cuda.empty_cache()

    np.savez_compressed("outputs/text_artifacts.npz", **artifacts)
    with open("text_artifacts_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved text_artifacts.npz and text_artifacts_summary.json")
    print("Downstream scripts can load axes without reloading any encoder.")

if __name__ == "__main__":
    main()