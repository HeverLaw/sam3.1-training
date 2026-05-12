"""
Merge a finetuned checkpoint back into a full SAM3.1 checkpoint.

Supports two modes:
  --mode tracker  (default) After training SAM3MultiplexTrain, merge finetuned
                  tracker weights back into the release checkpoint.
                  backbone.* -> detector.backbone.*, rest -> tracker.*

  --mode detector After training a detector-only pipeline, merge finetuned
                  detector weights back into the release checkpoint.
                  all keys -> detector.*

Usage:
    python training/utils/merge_checkpoint.py \
        --pretrained /path/to/sam3.1_release.pt \
        --finetuned /path/to/checkpoints/checkpoint.pt \
        --output /path/to/sam3.1_merged.pt \
        [--mode tracker|detector]
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
    mode: str = "tracker",
    finetuned_ckpt_key: str = "model",
    pretrained_ckpt_key: str = "model",
    verbose: bool = True,
):
    """
    Merge finetuned weights into the full SAM3.1 pretrained checkpoint.

    Args:
        pretrained_path: Path to official SAM3.1 release checkpoint containing
            both detector and tracker keys.
        finetuned_path: Path to the finetuned checkpoint.
        output_path: Where to save the merged checkpoint.
        mode: "tracker" for SAM3MultiplexTrain output, or "detector" for
            detector-only output.
        finetuned_ckpt_key: Key in finetuned checkpoint dict that holds the
            state_dict. Set to None if the checkpoint is already the state_dict.
        pretrained_ckpt_key: Same for pretrained checkpoint.
        verbose: Whether to print detailed merge info.
    """
    if mode not in {"tracker", "detector"}:
        raise ValueError(f"Unsupported merge mode: {mode}")

    logger.info(f"Loading pretrained checkpoint from {pretrained_path}")
    pretrained_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    if pretrained_ckpt_key and pretrained_ckpt_key in pretrained_ckpt:
        pretrained_sd = pretrained_ckpt[pretrained_ckpt_key]
    else:
        pretrained_sd = pretrained_ckpt

    logger.info(f"Loading finetuned checkpoint from {finetuned_path}")
    finetuned_ckpt = torch.load(finetuned_path, map_location="cpu", weights_only=True)
    if finetuned_ckpt_key and finetuned_ckpt_key in finetuned_ckpt:
        finetuned_sd = finetuned_ckpt[finetuned_ckpt_key]
    else:
        finetuned_sd = finetuned_ckpt

    uses_tracker_model_prefix = any(k.startswith("tracker.model.") for k in pretrained_sd)
    tracker_prefix = "tracker.model." if uses_tracker_model_prefix else "tracker."
    if verbose:
        print(f"  Detected tracker prefix in pretrained: '{tracker_prefix}'")

    new_state_dict = {}
    if mode == "detector":
        for k, v in finetuned_sd.items():
            new_state_dict["detector." + k] = v
    else:
        for k, v in finetuned_sd.items():
            if k.startswith("backbone"):
                new_state_dict["detector." + k] = v
            else:
                new_state_dict[tracker_prefix + k] = v
    finetuned_sd = new_state_dict

    overwritten_keys = set(finetuned_sd.keys()) & set(pretrained_sd.keys())
    new_keys = set(finetuned_sd.keys()) - set(pretrained_sd.keys())
    kept_keys = set(pretrained_sd.keys()) - set(finetuned_sd.keys())

    pretrained_sd.update(finetuned_sd)

    if verbose:
        component = "detector" if mode == "detector" else "tracker"
        print(f"Merge summary (mode={mode}):")
        print(f"  Pretrained keys (total):          {len(pretrained_sd)}")
        print(f"  Finetuned keys:                   {len(finetuned_sd)}")
        print(f"  Overwritten ({component} updated): {len(overwritten_keys)}")
        print(f"  New (only in finetuned):          {len(new_keys)}")
        print(f"  Kept (unchanged from pretrained): {len(kept_keys)}")
        if new_keys:
            print(f"  New keys not in pretrained (first 10): {sorted(new_keys)[:10]}")
        print(f"  Sample overwritten keys: {sorted(overwritten_keys)[:5]}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged_ckpt = {"model": pretrained_sd}
    for k in pretrained_ckpt:
        if k != pretrained_ckpt_key and k not in merged_ckpt:
            merged_ckpt[k] = pretrained_ckpt[k]
    torch.save(merged_ckpt, output_path)
    print(f"Merged checkpoint saved to {output_path}")

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Merge finetuned weights into full SAM3.1 checkpoint")
    parser.add_argument("--pretrained", required=True, help="Path to SAM3.1 release checkpoint")
    parser.add_argument("--finetuned", required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged checkpoint")
    parser.add_argument(
        "--mode",
        default="tracker",
        choices=["tracker", "detector"],
        help="tracker: SAM3MultiplexTrain output; detector: detector-only output",
    )
    args = parser.parse_args()
    merge_tracker_into_full_ckpt(
        args.pretrained,
        args.finetuned,
        args.output,
        mode=args.mode,
    )
