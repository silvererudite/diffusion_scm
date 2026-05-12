"""
03c_multi_encoder_axes.py
-------------------------
Build W/C text axes from the SD3-medium text encoders, one at a time.

SD3-medium conditions on THREE text encoders in parallel:
  - CLIP ViT-L/14         (768-D)   handled in 03b_fix_clip_axes.py
  - OpenCLIP ViT-bigG/14  (1280-D)  this script
  - T5-XXL                (4096-D)  this script (optional, gated by flag)

Each encoder gets bipolar (positive minus negative pole) axes — the unipolar
mean-of-positives approach collapses because L2-normalized embeddings share
a generic "photo of a person" prior.

We do NOT project image embeddings onto these text axes — that's a modality
mismatch. Instead, each encoder's axes are evaluated on its own terms by
projecting ITS OWN text embeddings of the 66 group names ("a photo of a CEO")
and correlating the resulting (W, C) coords with the human ground truth.

This answers: "Does this text encoder, on its own, contain SCM structure
sufficient to recover human warmth/competence ratings of social groups?"

The image-side probe (DINOv2) remains the answer to "does SD3's OUTPUT
express SCM structure" — handled separately.

Outputs:
  text_axis_probe_bigG.csv   (per-group W,C coords + correlations to ground truth)
  text_axis_probe_t5.csv     (if --include_t5)

Usage:
  python 03c_multi_encoder_axes.py                # bigG only
  python 03c_multi_encoder_axes.py --include_t5   # both
"""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

WARMTH_POS = ["warm", "friendly", "sincere", "trustworthy", "kind", "caring",
              "honest", "good-natured", "tolerant", "sociable", "generous",
              "compassionate", "approachable"]
WARMTH_NEG = ["cold", "unfriendly", "insincere", "untrustworthy", "cruel",
              "callous", "dishonest", "mean", "intolerant", "antisocial",
              "selfish", "hostile", "unapproachable"]
COMPETENCE_POS = ["competent", "skilled", "intelligent", "capable", "efficient",
                  "confident", "knowledgeable", "clever", "independent",
                  "ambitious", "successful", "professional", "accomplished"]
COMPETENCE_NEG = ["incompetent", "unskilled", "unintelligent", "incapable",
                  "inefficient", "insecure", "ignorant", "foolish", "dependent",
                  "unambitious", "unsuccessful", "unprofessional", "failed"]

PROMPT_TEMPLATE = "a photo of a {} person"
GROUP_TEMPLATE  = "a photo of a {}"

def build_bipolar_axis(encode_fn, pos_words, neg_words):
    """Encode pos and neg word lists, L2-normalize, average per pole, subtract."""
    pos = encode_fn([PROMPT_TEMPLATE.format(w) for w in pos_words])
    neg = encode_fn([PROMPT_TEMPLATE.format(w) for w in neg_words])
    pos = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    neg = neg / np.linalg.norm(neg, axis=-1, keepdims=True)
    axis = pos.mean(0) - neg.mean(0)
    return axis / np.linalg.norm(axis)

def evaluate_encoder(name, encode_fn, gt):
    """Build bipolar W/C axes, project group-name embeddings, correlate with ground truth."""
    print(f"\n{'='*60}\n  Encoder: {name}\n{'='*60}")

    v_w = build_bipolar_axis(encode_fn, WARMTH_POS, WARMTH_NEG)
    v_c = build_bipolar_axis(encode_fn, COMPETENCE_POS, COMPETENCE_NEG)
    angle = np.degrees(np.arccos(np.clip(v_w @ v_c, -1, 1)))
    print(f"  W-C axis angle: {angle:.1f}°  (90° = orthogonal, ideal)")

    # Embed each group name once, project onto axes
    group_keys = gt["category_key"].tolist()
    group_prompts = [GROUP_TEMPLATE.format(k) for k in group_keys]
    grp_emb = encode_fn(group_prompts)
    grp_emb = grp_emb / np.linalg.norm(grp_emb, axis=-1, keepdims=True)
    w_pred = grp_emb @ v_w
    c_pred = grp_emb @ v_c

    # Correlate with ground truth (z-scored within source already)
    w_r, w_p = pearsonr(w_pred, gt["warmth_z"])
    c_r, c_p = pearsonr(c_pred, gt["competence_z"])
    w_rho, _ = spearmanr(w_pred, gt["warmth_z"])
    c_rho, _ = spearmanr(c_pred, gt["competence_z"])

    print(f"  Warmth      Pearson r = {w_r:+.3f}  (p={w_p:.2e})   Spearman ρ = {w_rho:+.3f}")
    print(f"  Competence  Pearson r = {c_r:+.3f}  (p={c_p:.2e})   Spearman ρ = {c_rho:+.3f}")

    # Show top-5 / bottom-5 for spot-checking
    df = pd.DataFrame({
        "category_key": group_keys,
        "w_pred": w_pred, "c_pred": c_pred,
        "w_gt": gt["warmth_z"].values, "c_gt": gt["competence_z"].values,
    })
    print(f"\n  Highest predicted competence:")
    print(df.nlargest(5, "c_pred")[["category_key","c_pred","c_gt"]].to_string(index=False))
    print(f"\n  Lowest predicted competence:")
    print(df.nsmallest(5, "c_pred")[["category_key","c_pred","c_gt"]].to_string(index=False))

    return df, {"angle": angle, "w_r": w_r, "c_r": c_r, "w_rho": w_rho, "c_rho": c_rho}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_t5", action="store_true",
                    help="Also probe T5-XXL (slow, ~20GB to download)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gt = pd.read_csv("outputs/groundtruth_clean.csv")
    # Restrict to categories that are also in the image dataset
    emb = np.load("outputs/embeddings.npz", allow_pickle=True)
    img_cats = set(np.unique(emb["categories"]).tolist())
    gt = gt[gt["category_key"].isin(img_cats)].reset_index(drop=True)
    print(f"Evaluating on {len(gt)} groups present in both groundtruth and dataset")

    # ============== OpenCLIP ViT-bigG/14 ==============
    print("\nLoading OpenCLIP ViT-bigG/14 (this is ~5GB)...")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

    def bigg_encode(texts):
        with torch.no_grad():
            tok = tokenizer(texts).to(args.device)
            feats = model.encode_text(tok)
        return feats.cpu().numpy().astype(np.float32)

    df_bigg, stats_bigg = evaluate_encoder("OpenCLIP-bigG", bigg_encode, gt)
    df_bigg.to_csv("text_axis_probe_bigG.csv", index=False)
    del model, tokenizer
    torch.cuda.empty_cache()

    # ============== T5-XXL (optional) ==============
    if args.include_t5:
        from load_t5_fp8 import load_t5_xxl_fp8, t5_encode_factory
        # Free bigG memory before loading T5 (T5 needs ~20GB GPU)
        torch.cuda.empty_cache()
        
        t5_model, t5_tok = load_t5_xxl_fp8(args.device)
        t5_encode = t5_encode_factory(t5_model, t5_tok, args.device)
        df_t5, stats_t5 = evaluate_encoder("T5-XXL (SD3 fp8 → bf16)", t5_encode, gt)
        df_t5.to_csv("text_axis_probe_t5.csv", index=False)

        del t5_model, t5_tok
        torch.cuda.empty_cache()

    # Compare with the CLIP-L result from 03b for context
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  OpenCLIP-bigG: W angle-with-C={stats_bigg['angle']:.0f}°, "
          f"r_W={stats_bigg['w_r']:+.2f}, r_C={stats_bigg['c_r']:+.2f}")
    if args.include_t5:
        print(f"  T5-XXL:        W angle-with-C={stats_t5['angle']:.0f}°, "
              f"r_W={stats_t5['w_r']:+.2f}, r_C={stats_t5['c_r']:+.2f}")
    print(f"\nInterpretation guide:")
    print(f"  |r| > 0.5 = encoder strongly contains the axis")
    print(f"  |r| 0.3-0.5 = present but noisy")
    print(f"  |r| < 0.3 = encoder is not a useful probe for that axis")

if __name__ == "__main__":
    main()