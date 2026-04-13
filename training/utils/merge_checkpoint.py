"""
Merge a finetuned tracker checkpoint back into a full SAM3.1 checkpoint.

After training SAM3MultiplexTrain (which only contains the tracker), this tool
merges the finetuned tracker weights back into the official SAM3.1 release
checkpoint (which contains both detector and tracker), producing a merged
checkpoint that supports the full open-vocabulary pipeline.

Usage:
    # Standalone
    python training/utils/merge_checkpoint.py \
        --pretrained /path/to/sam3.1_release.pt \
        --finetuned /path/to/your_finetuned/checkpoints/checkpoint.pt \
        --output /path/to/sam3.1_merged.pt

    # Or call from Python
    from training.utils.merge_checkpoint import merge_tracker_into_full_ckpt
    merge_tracker_into_full_ckpt(pretrained_path, finetuned_path, output_path)
"""

import logging
import os
from argparse import ArgumentParser

import torch

logger = logging.getLogger(__name__)


def merge_tracker_into_full_ckpt(
    pretrained_path: str,
    finetuned_path: str,
    output_path: str,
    finetuned_ckpt_key: str = "model",
    pretrained_ckpt_key: str = "model",
    verbose: bool = True,
):
    """
    Merge finetuned tracker weights into the full SAM3.1 pretrained checkpoint.

    Args:
        pretrained_path: Path to official SAM3.1 release checkpoint (contains
            both detector and tracker keys).
        finetuned_path: Path to the finetuned tracker checkpoint produced by
            SAM3MultiplexTrain training (contains only tracker keys).
        output_path: Where to save the merged checkpoint.
        finetuned_ckpt_key: Key in finetuned checkpoint dict that holds the
            state_dict (e.g. "model"). Set to None if the ckpt IS the state_dict.
        pretrained_ckpt_key: Same for pretrained checkpoint.
        verbose: Whether to print detailed merge info.
    """
    # --- Load pretrained (full model: detector + tracker) ---
    logger.info(f"Loading pretrained checkpoint from {pretrained_path}")
    pretrained_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    if pretrained_ckpt_key and pretrained_ckpt_key in pretrained_ckpt:
        pretrained_sd = pretrained_ckpt[pretrained_ckpt_key]
    else:
        pretrained_sd = pretrained_ckpt

    # --- Load finetuned (tracker only) ---
    logger.info(f"Loading finetuned checkpoint from {finetuned_path}")
    finetuned_ckpt = torch.load(finetuned_path, map_location="cpu", weights_only=True)
    if finetuned_ckpt_key and finetuned_ckpt_key in finetuned_ckpt:
        finetuned_sd = finetuned_ckpt[finetuned_ckpt_key]
    else:
        finetuned_sd = finetuned_ckpt

    new_state_dict = {}
    for k, v in finetuned_sd.items():
        if k.startswith("backbone"):
            # 映射回 detector 分支
            new_state_dict["detector." + k] = v
        else:
            # 映射回 tracker 分支
            new_state_dict["tracker.model." + k] = v
    finetuned_sd = new_state_dict

    # --- Merge: finetuned tracker keys overwrite pretrained tracker keys ---
    overwritten_keys = set(finetuned_sd.keys()) & set(pretrained_sd.keys())
    new_keys = set(finetuned_sd.keys()) - set(pretrained_sd.keys())
    kept_keys = set(pretrained_sd.keys()) - set(finetuned_sd.keys())

    pretrained_sd.update(finetuned_sd)

    if verbose:
        print(f"Merge summary:")
        print(f"  Pretrained keys (total):       {len(pretrained_sd)}")
        print(f"  Finetuned keys:                {len(finetuned_sd)}")
        print(f"  Overwritten (tracker updated): {len(overwritten_keys)}")
        print(f"  New (only in finetuned):       {len(new_keys)}")
        print(f"  Kept (detector, unchanged):    {len(kept_keys)}")
        if new_keys:
            print(f"  New keys not in pretrained (first 10): {sorted(new_keys)[:10]}")
        # Print a few overwritten keys as sanity check
        sample = sorted(overwritten_keys)[:5]
        print(f"  Sample overwritten keys: {sample}")

    # --- Save ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged_ckpt = {"model": pretrained_sd}
    # Preserve any non-model metadata from pretrained ckpt
    for k in pretrained_ckpt:
        if k != pretrained_ckpt_key and k not in merged_ckpt:
            merged_ckpt[k] = pretrained_ckpt[k]
    torch.save(merged_ckpt, output_path)
    print(f"Merged checkpoint saved to {output_path}")

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Merge finetuned tracker into full SAM3.1 checkpoint")
    parser.add_argument("--pretrained", required=True, help="Path to SAM3.1 release checkpoint")
    parser.add_argument("--finetuned", required=True, help="Path to finetuned tracker checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged checkpoint")
    args = parser.parse_args()
    merge_tracker_into_full_ckpt(args.pretrained, args.finetuned, args.output)