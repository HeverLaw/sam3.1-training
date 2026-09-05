# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Adapter for training SAM3.1 VideoTrackingMultiplex with SAM2's training infra.

"""
SAM3MultiplexTrain: inherits VideoTrackingDynamicMultiplex and overrides
forward() to accept SAM2's BatchedVideoDatapoint — so SAM2's trainer /
dataset / loss are reused zero-change.

State dict is 1:1 compatible with SAM3.1 official checkpoints because we
inherit (not wrap), so keys are `backbone.*`, `sam_mask_decoder.*` etc.
without any prefix.

Usage in Hydra config:
  model:
    _target_: training.model.sam3_multiplex_train.SAM3MultiplexTrain
    image_size: 672
    freeze_image_encoder: true
    ...
"""

import fnmatch
import logging
from typing import List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex
from sam3.model.data_misc import NestedTensor
from sam3.model.multiplex_utils import MultiplexController
from sam3.model.vl_combiner import TriHeadVisionOnly
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from sam3.model.model_misc import TransformerWrapper
from sam3.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from sam3.model.necks import Sam3TriViTDetNeck
from sam3.model.vitdet import ViT

from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

logger = logging.getLogger(__name__)


class SAM3MultiplexTrain(VideoTrackingDynamicMultiplex):
    """
    Inherits VideoTrackingDynamicMultiplex and adds:
    1. Model construction from SAM3.1 official builder functions
    2. forward() adapter: SAM2 BatchedVideoDatapoint -> SAM3.1 forward
    3. Checkpoint loading + selective freezing
    4. State dict 1:1 compatible with SAM3.1 (no prefix wrapping)
    """

    def __init__(
        self,
        # --- architecture ---
        image_size: int = 672,
        multiplex_count: int = 16,
        num_maskmem: int = 7,
        # --- training strategy (passed to parent) ---
        prob_to_use_pt_input_for_train: float = 0.0,
        prob_to_use_pt_input_for_eval: float = 0.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        num_frames_to_correct_for_train: int = 1,
        num_frames_to_correct_for_eval: int = 1,
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        num_init_cond_frames_for_train: int = 1,
        num_init_cond_frames_for_eval: int = 1,
        rand_init_cond_frames_for_train: bool = True,
        rand_init_cond_frames_for_eval: bool = False,
        add_all_frames_to_correct_as_cond: bool = False,
        num_correction_pt_per_frame: int = 7,
        pt_sampling_for_eval: str = "center",
        prob_to_sample_from_gt_for_train: float = 0.0,
        # --- freeze ---
        freeze_image_encoder: bool = False,
        freeze_patterns: Optional[List[str]] = None,
        # --- checkpoint ---
        checkpoint_path: Optional[str] = None,
        # --- backbone compile ---
        compile_image_encoder: bool = False,
        use_fa3: bool = False,
        use_rope_real: bool = False,
        # --- eval ---
        forward_backbone_per_frame_for_eval: bool = False,
        # --- dynamic object admission augmentation ---
        dynamic_object_delay_prob: float = 0.0,  # legacy option; nonzero is rejected
        prob_to_dropout_spatial_mem: float = 0.0,  # prob to dropout spatial mem for a random frame during training (0=off)
        prob_condition_all_objects_on_init_for_train: float = 1.0,
        ratio_of_objects_to_condition_on_init_for_train: float = 1.0,
        rand_objects_to_condition_on_init_for_train: bool = True,
    ):
        for name, value in {
            "prob_condition_all_objects_on_init_for_train": prob_condition_all_objects_on_init_for_train,
            "ratio_of_objects_to_condition_on_init_for_train": ratio_of_objects_to_condition_on_init_for_train,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if dynamic_object_delay_prob != 0:
            raise ValueError(
                "dynamic_object_delay_prob is no longer supported: it erased physical GT. "
                "Use prob_condition_all_objects_on_init_for_train instead."
            )
        if (
            prob_condition_all_objects_on_init_for_train < 1.0
            and num_init_cond_frames_for_train != 1
        ):
            raise ValueError("Partial initial admission requires num_init_cond_frames_for_train=1")

        # ── Build components (resolution-aware, derived from model_builder.py) ──
        backbone_stride = 14
        feat_size = image_size // backbone_stride  # 1008→72, 672→48
        # mask downsampler interpol size: roughly image_size * 8/7 rounded
        interpol_size = [int(image_size * 8 / 7)] * 2

        maskmem_backbone = _build_multiplex_maskmem_backbone(
            multiplex_count=multiplex_count,
            precompute_resolution=image_size,
            interpol_size=interpol_size,
        )
        transformer = _build_multiplex_transformer(
            feat_size=feat_size,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )
        tri_neck = _build_multiplex_tri_backbone(
            image_size=image_size,
            compile_mode="max-autotune" if compile_image_encoder else None,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )
        backbone = TriHeadVisionOnly(
            visual=tri_neck, n_features=256, scalp=0
        )
        multiplex_controller = MultiplexController(
            multiplex_count=multiplex_count,
            eval_multiplex_count=multiplex_count,
        )

        # ── Call parent __init__ with SAM3.1 official defaults ──
        # (from model_builder.py build_sam3_multiplex_video_model L985-L1047)
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            multiplex_controller=multiplex_controller,
            image_size=image_size,
            backbone_stride=14,
            num_maskmem=num_maskmem,
            # prompt / correction
            prob_to_use_pt_input_for_train=prob_to_use_pt_input_for_train,
            prob_to_use_pt_input_for_eval=prob_to_use_pt_input_for_eval,
            prob_to_use_box_input_for_train=prob_to_use_box_input_for_train,
            prob_to_use_box_input_for_eval=prob_to_use_box_input_for_eval,
            num_frames_to_correct_for_train=num_frames_to_correct_for_train,
            num_frames_to_correct_for_eval=num_frames_to_correct_for_eval,
            rand_frames_to_correct_for_train=rand_frames_to_correct_for_train,
            rand_frames_to_correct_for_eval=rand_frames_to_correct_for_eval,
            num_init_cond_frames_for_train=num_init_cond_frames_for_train,
            num_init_cond_frames_for_eval=num_init_cond_frames_for_eval,
            rand_init_cond_frames_for_train=rand_init_cond_frames_for_train,
            rand_init_cond_frames_for_eval=rand_init_cond_frames_for_eval,
            add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
            num_correction_pt_per_frame=num_correction_pt_per_frame,
            pt_sampling_for_eval=pt_sampling_for_eval,
            prob_to_sample_from_gt_for_train=prob_to_sample_from_gt_for_train,
            # SAM3.1 official model defaults
            use_high_res_features_in_sam=True,
            use_obj_ptrs_in_encoder=True,
            max_obj_ptrs_in_encoder=16,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_mlp_for_obj_ptr_proj=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            use_no_obj_ptr=True,
            use_linear_no_obj_ptr=True,
            no_obj_embed_spatial=True,
            sincos_tpos_enc=True,
            multimask_output_in_sam=True,
            multimask_output_for_tracking=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_multimask_token_for_obj_ptr=True,
            num_multimask_outputs=3,
            apply_sigmoid_to_mask_logits_for_mem_enc=True,
            sigmoid_scale_for_mem_enc=2.0,
            sigmoid_bias_for_mem_enc=-1.0,
            non_overlap_masks_for_mem_enc=False,
            add_output_suppression_embeddings=True,
            add_object_conditional_embeddings=False,
            condition_as_mask_input=True,
            condition_as_mask_input_fg=1.0,
            condition_as_mask_input_bg=0.0,
            use_maskmem_tpos_v2=True,
            save_image_features=True,
            randomness_fix=True,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            iou_prediction_use_sigmoid=False,
            forward_backbone_per_frame_for_eval=forward_backbone_per_frame_for_eval,
            offload_output_to_cpu_for_eval=False,
            trim_past_non_cond_mem_for_eval=False,
            max_cond_frames_in_attn=4,
            is_dynamic_model=True,
            sam_mask_decoder_extra_args={
                "dynamic_multimask_via_stability": True,
                "dynamic_multimask_stability_delta": 0.05,
                "dynamic_multimask_stability_thresh": 0.98,
            },
            compile_all_components=False,
            use_memory_selection=False,
            share_necks=False,
        )

        self.prob_to_dropout_spatial_mem = prob_to_dropout_spatial_mem  # TODO: should be used in the multiplex training

        # ── Dynamic object admission augmentation ──
        self.prob_condition_all_objects_on_init_for_train = prob_condition_all_objects_on_init_for_train
        self.ratio_of_objects_to_condition_on_init_for_train = ratio_of_objects_to_condition_on_init_for_train
        self.rand_objects_to_condition_on_init_for_train = rand_objects_to_condition_on_init_for_train
        self.checkpoint_path = checkpoint_path

        # ── Load checkpoint ──
        # Because we inherit (not wrap), self.load_state_dict() uses keys
        # directly matching SAM3.1 format: backbone.*, sam_mask_decoder.*, etc.
        if checkpoint_path:
            self._load_sam3_checkpoint(checkpoint_path)

        # ── Freeze ──
        if freeze_image_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("[SAM3MultiplexTrain] Froze entire image encoder (self.backbone)")

        if freeze_patterns:
            self._apply_freeze_patterns(freeze_patterns)

    # ─────────────────────────────────────────────────────────
    #  Forward: SAM2 BatchedVideoDatapoint -> list[dict]
    # ─────────────────────────────────────────────────────────
    def forward(self, input: BatchedVideoDatapoint):
        """
        Override forward to accept SAM2's BatchedVideoDatapoint.
        Adapts it internally, then calls parent's prepare_prompt_inputs
        + forward_tracking which are training-ready.

        Supports batch_size > 1 by processing each video independently.
        Returns list of (per_video_outputs, per_video_masks) when batch > 1,
        so the trainer can compute loss per video and average.
        """
        num_videos = input.num_videos

        if num_videos == 1:
            outputs, updated_targets = self._forward_single_video(input)
            # Always return (outputs, targets) tuple so Trainer uses the
            # dynamic-multiplex-adjusted targets (which may have fewer objects
            # on transition-point frames) instead of the original batch.masks.
            return [(outputs, updated_targets)]

        # Multi-video batch: return per-video results for separate loss computation
        results = []
        for vid_idx in range(num_videos):
            single_input = _split_single_video(input, vid_idx)
            outputs, updated_targets = self._forward_single_video(single_input)
            results.append((outputs, updated_targets))
        return results

    def _forward_single_video(self, input: BatchedVideoDatapoint):
        """Forward for a single video (batch_size=1)."""
        # 1. Compute image features
        # Split into two stages:
        #   a) Backbone forward (ViT) — may need torch.no_grad() when frozen
        #      to avoid addmm_act ValueError
        #   b) conv_s0/conv_s1 projections on mask decoders — these are trainable
        #      and must NOT be inside no_grad
        flat_imgs = input.flat_img_batch  # [T, C, H, W]
        img_nested = NestedTensor(tensors=flat_imgs, mask=None)

        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
            if backbone_frozen:
                # Run backbone (ViT) without grad, but conv_s0/conv_s1 need grad
                with torch.no_grad():
                    backbone_out = self.backbone.forward_image(
                        img_nested,
                        need_interactive_out=True,
                        need_propagation_out=True,
                    )
                # Run the high-res feature projections (conv_s0/conv_s1) WITH grad
                # These are trainable parameters on the mask decoders
                self._apply_high_res_feature_projections(backbone_out)
            else:
                backbone_out = self.forward_image(
                    img_nested,
                    need_interactive_out=True,
                    need_propagation_out=True,
                )
        else:
            backbone_out = {}

        # 2. Adapt SAM2 batch to SAM3 interface and prepare prompts
        adapted_input = _SAM2ToSAM3InputAdapter(input)
        backbone_out = self.prepare_prompt_inputs(backbone_out, adapted_input)

        # 3. Run forward_tracking (inherits from VideoTrackingDynamicMultiplex)
        all_frame_outputs = self.forward_tracking(
            backbone_out, adapted_input, return_dict=False
        )

        # 4. Rebuild targets from adapted_input.find_targets (which was modified
        #    in-place by prepare_prompt_inputs for dynamic multiplex — e.g. fewer
        #    objects on transition-point frames).
        # Use list (not torch.stack) because dynamic multiplex may produce
        # different numbers of objects per frame at transition points.
        updated_targets = [
            adapted_input.find_targets[t].segments
            for t in range(len(adapted_input.find_targets))
        ]

        return all_frame_outputs, updated_targets

    def _prepare_object_admission(self, backbone_out, input, start_frame_idx):
        if not self.training:
            return
        if not self.enable_dynamic_training and (
            self.prob_condition_all_objects_on_init_for_train < 1.0
        ):
            raise ValueError("Object admission augmentation requires dynamic training")
        input.prepare_object_admission(
            init_cond_frames=backbone_out["init_cond_frames"],
            start_frame_idx=start_frame_idx,
            prob_all=self.prob_condition_all_objects_on_init_for_train,
            ratio=self.ratio_of_objects_to_condition_on_init_for_train,
            random_count=self.rand_objects_to_condition_on_init_for_train,
            rng=self.rng2,
        )

    def _apply_high_res_feature_projections(self, backbone_out):
        """
        Apply conv_s0/conv_s1 projections and clone tensors — the parts of
        forward_image that involve trainable mask-decoder parameters.
        Must be called outside torch.no_grad() when backbone is frozen.
        """
        if self.use_high_res_features_in_sam:
            if "interactive" in backbone_out:
                backbone_out["interactive"]["backbone_fpn"][
                    0
                ].tensors = self.interactive_sam_mask_decoder.conv_s0(
                    backbone_out["interactive"]["backbone_fpn"][0].tensors
                )
                backbone_out["interactive"]["backbone_fpn"][
                    1
                ].tensors = self.interactive_sam_mask_decoder.conv_s1(
                    backbone_out["interactive"]["backbone_fpn"][1].tensors
                )
            if "sam2_backbone_out" in backbone_out:
                backbone_out["sam2_backbone_out"]["backbone_fpn"][
                    0
                ].tensors = self.sam_mask_decoder.conv_s0(
                    backbone_out["sam2_backbone_out"]["backbone_fpn"][0].tensors
                )
                backbone_out["sam2_backbone_out"]["backbone_fpn"][
                    1
                ].tensors = self.sam_mask_decoder.conv_s1(
                    backbone_out["sam2_backbone_out"]["backbone_fpn"][1].tensors
                )
        # Clone to help torch.compile
        from sam3.model.video_tracking_multiplex import neck_outs
        for out_type in neck_outs:
            if out_type not in backbone_out:
                continue
            for i in range(len(backbone_out[out_type]["backbone_fpn"])):
                backbone_out[out_type]["backbone_fpn"][i].tensors = self._maybe_clone(
                    backbone_out[out_type]["backbone_fpn"][i].tensors
                )
                backbone_out[out_type]["vision_pos_enc"][i] = self._maybe_clone(
                    backbone_out[out_type]["vision_pos_enc"][i]
                )

    # ─────────────────────────────────────────────────────────
    #  Checkpoint loading (SAM3.1 compatible keys)
    # ─────────────────────────────────────────────────────────
    def _load_sam3_checkpoint(self, ckpt_path: str):
        logger.info(f"[SAM3MultiplexTrain] Loading checkpoint from {ckpt_path}")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        # Official sam3.1_multiplex.pt checkpoint structure:
        #   tracker.model.backbone.*    → tracker's backbone (deleted in predictor build, keys absent)
        #   tracker.model.<other>.*     → tracker's other params (sam_mask_decoder, transformer, etc.)
        #   detector.backbone.*         → detector's backbone (contains the ViT weights)
        #   detector.<other>.*          → detector's other params (not needed)
        #
        # This model (SAM3MultiplexTrain) inherits VideoTrackingDynamicMultiplex directly,
        # so its keys are "backbone.*", "sam_mask_decoder.*", etc.
        #
        # We remap:
        #   tracker.model.<key>  → <key>          (tracker params)
        #   detector.backbone.<key> → backbone.<key>  (backbone from detector, since tracker's was deleted)
        has_tracker_model_prefix = any(k.startswith("tracker.model.") for k in ckpt)
        if has_tracker_model_prefix:
            remapped = {}
            tracker_prefix = "tracker.model."
            det_backbone_prefix = "detector.backbone."
            for k, v in ckpt.items():
                if k.startswith(tracker_prefix):
                    remapped[k[len(tracker_prefix):]] = v
                elif k.startswith(det_backbone_prefix):
                    new_key = "backbone." + k[len(det_backbone_prefix):]
                    # only use detector backbone if tracker didn't provide it
                    if new_key not in remapped:
                        remapped[new_key] = v
            ckpt = remapped
            print(f"[SAM3MultiplexTrain] Remapped checkpoint keys: {len(ckpt)} keys total")

        # Filter out keys with shape mismatch (e.g. freqs_cis when resolution differs)
        model_state = self.state_dict()
        filtered_ckpt = {}
        skipped = []
        ignored = []
        for k, v in ckpt.items():
            if k not in model_state:
                ignored.append(k)
            elif v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
            else:
                filtered_ckpt[k] = v
        if ignored:
            print(
                f"[SAM3MultiplexTrain] Ignored {len(ignored)} checkpoint-only keys "
                f"(first 10: {ignored[:10]})"
            )
        if skipped:
            print(f"[SAM3MultiplexTrain] Skipped {len(skipped)} keys with shape mismatch:")
            for s in skipped:
                print(f"  {s}")
        missing, unexpected = self.load_state_dict(filtered_ckpt, strict=False)
        print(f"[SAM3MultiplexTrain] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[SAM3MultiplexTrain] first 10 missing: {missing[:10]}")
        if unexpected:
            print(f"[SAM3MultiplexTrain] first 10 unexpected: {unexpected[:10]}")

    # ─────────────────────────────────────────────────────────
    #  Freeze
    # ─────────────────────────────────────────────────────────
    def _apply_freeze_patterns(self, freeze_patterns: List[str]):
        n_frozen = 0
        for name, p in self.named_parameters():
            if any(fnmatch.fnmatchcase(name, pat) for pat in freeze_patterns):
                p.requires_grad = False
                n_frozen += 1
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[SAM3MultiplexTrain] freeze_patterns froze {n_frozen} params "
            f"(trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M = {100*trainable/total:.1f}%)"
        )


# ═══════════════════════════════════════════════════════════
#  Multi-video batch splitting
# ═══════════════════════════════════════════════════════════

def _split_single_video(batch: BatchedVideoDatapoint, vid_idx: int) -> BatchedVideoDatapoint:
    """
    Extract a single video from a multi-video batch.

    BatchedVideoDatapoint shapes (B videos, T frames, O_total objects):
        img_batch:        [T, B, C, H, W]
        masks:            [T, O_total, H, W]
        obj_to_frame_idx: [T, O_total, 2]  — last dim is (frame_idx, video_idx)
        metadata.unique_objects_identifier: [T, O_total, 3]
        metadata.frame_orig_size:           [T, O_total, 2]

    Returns a BatchedVideoDatapoint with B=1 and only this video's objects.
    """
    T = batch.num_frames
    device = batch.masks.device

    # 1. Extract single video's images: [T, B, C, H, W] → [T, 1, C, H, W]
    img_single = batch.img_batch[:, vid_idx : vid_idx + 1]

    # 2. Find which objects belong to this video (use frame 0 as reference)
    #    obj_to_frame_idx[t, o, 1] == video_idx
    video_indices = batch.obj_to_frame_idx[0, :, 1]  # [O_total]
    obj_mask = video_indices == vid_idx  # [O_total] bool
    obj_indices = torch.where(obj_mask)[0]

    # 3. Slice masks and obj_to_frame_idx for this video's objects
    masks_single = batch.masks[:, obj_indices]  # [T, O_vid, H, W]

    # 4. Build new obj_to_frame_idx: video_idx is now 0
    obj_to_frame_single = batch.obj_to_frame_idx[:, obj_indices].clone()  # [T, O_vid, 2]
    obj_to_frame_single[:, :, 1] = 0  # remap video_idx to 0

    # 5. Slice metadata
    meta_ids = batch.metadata.unique_objects_identifier[:, obj_indices]
    meta_sizes = batch.metadata.frame_orig_size[:, obj_indices]
    metadata_single = BatchedVideoMetaData(
        unique_objects_identifier=meta_ids,
        frame_orig_size=meta_sizes,
        batch_size=[T],
    )

    return BatchedVideoDatapoint(
        img_batch=img_single,
        obj_to_frame_idx=obj_to_frame_single,
        masks=masks_single,
        metadata=metadata_single,
        dict_key=batch.dict_key,
        batch_size=[T],
    )


# ═══════════════════════════════════════════════════════════
#  Input adapter: SAM2 BatchedVideoDatapoint -> SAM3 interface
# ═══════════════════════════════════════════════════════════

class _FindStageProxy:
    """Duck-type proxy for SAM3's FindStage — only img_ids is needed."""
    def __init__(self, img_ids: torch.Tensor):
        self.img_ids = img_ids


class _FindTargetProxy:
    """Duck-type proxy for SAM3's BatchedFindTarget."""
    def __init__(self, segments: torch.Tensor, num_boxes: torch.Tensor):
        self.segments = segments
        self.num_boxes = num_boxes


class _SAM2ToSAM3InputAdapter:
    """
    Minimal adapter making SAM2's BatchedVideoDatapoint look like SAM3's
    BatchedDatapoint — just enough for VideoTrackingMultiplex.prepare_prompt_inputs
    and forward_tracking to work.

    SAM2 data shapes:
        input.img_batch:           [T, B, C, H, W]
        input.flat_img_batch:      [(B*T), C, H, W]
        input.masks:               [T, O, H, W]   (T=num_frames, O=num_objects)
        input.flat_obj_to_img_idx: [T, O]          flat index into flat_img_batch

    SAM3 expects:
        input.img_batch:           NestedTensor(tensors=[(B*T), C, H, W], mask=None)
        input.find_inputs[t]:      .img_ids -> [O] tensor
        input.find_targets[t]:     .segments -> [O, H, W], .num_boxes -> [O]
        input.visible_objects_per_frame: dict[int, set[int]]  (for dynamic multiplex)
    """

    def __init__(self, sam2_batch: BatchedVideoDatapoint):
        self.img_batch = NestedTensor(
            tensors=sam2_batch.flat_img_batch, mask=None
        )

        num_frames = sam2_batch.num_frames
        num_objects = sam2_batch.masks.shape[1]
        device = sam2_batch.masks.device
        # Immutable-by-contract physical targets, before official reordering/slicing.
        # Keeping references is sufficient: admission never writes mask pixels.
        self.physical_masks = sam2_batch.masks

        # find_inputs: per-frame img_ids
        self.find_inputs = []
        for t in range(num_frames):
            img_ids = sam2_batch.flat_obj_to_img_idx[t]  # [O]
            self.find_inputs.append(_FindStageProxy(img_ids))

        # find_targets: per-frame GT masks
        self.find_targets = []
        for t in range(num_frames):
            seg = sam2_batch.masks[t]  # [O, H, W] bool
            num_boxes = torch.ones(seg.shape[0], device=device)
            self.find_targets.append(_FindTargetProxy(seg, num_boxes))

        # visible_objects_per_frame: for dynamic multiplex training
        self.visible_objects_per_frame = {}
        for t in range(num_frames):
            mask_t = sam2_batch.masks[t]  # [O, H, W]
            visible = set()
            for obj_idx in range(num_objects):
                if mask_t[obj_idx].any():
                    visible.add(obj_idx)
            self.visible_objects_per_frame[t] = visible

        self.physical_visible_objects_per_frame = {
            t: frozenset(ids) for t, ids in self.visible_objects_per_frame.items()
        }

    def prepare_object_admission(
        self,
        *,
        init_cond_frames,
        start_frame_idx,
        prob_all,
        ratio,
        random_count,
        rng,
    ):
        """Restrict discovery, never physical presence or target mask pixels.

        Called after the official initial-frame sampler and before its object union.
        A discoverable object is admitted only at an official sampled transition.
        It can remain unknown for the entire clip if no transition sees it.
        """
        physical = self.physical_visible_objects_per_frame
        if not physical[start_frame_idx]:
            # The upstream empty-start fallback zeroes object 0 across the clip.
            # Reject this sample explicitly rather than corrupting physical GT.
            raise ValueError(
                "Empty initial frame after transforms; resample the clip instead of zeroing GT"
            )
        self.visible_objects_per_frame = {t: set(ids) for t, ids in physical.items()}
        if prob_all == 1.0:
            return
        if len(init_cond_frames) != 1 or init_cond_frames[0] != start_frame_idx:
            raise ValueError(
                "Object admission augmentation supports one selected initial frame only"
            )

        init_frame = init_cond_frames[0]
        init_objects = sorted(physical[init_frame])
        # A static-image episode has no opportunity for subsequent admission.
        if len(physical) - init_frame <= 1:
            return
        selected = set(init_objects)
        if len(init_objects) > 1 and rng.random() >= prob_all:
            max_num = min(len(init_objects) - 1, max(1, int(len(init_objects) * ratio)))
            count = int(rng.integers(1, max_num + 1)) if random_count else max_num
            selected = set(rng.choice(init_objects, size=count, replace=False).tolist())
        self.visible_objects_per_frame[init_frame] = selected


# ═══════════════════════════════════════════════════════════
#  Resolution-aware builder functions
#  (derived from model_builder.py, but parameterized instead of hardcoded)
# ═══════════════════════════════════════════════════════════

def _build_multiplex_maskmem_backbone(
    multiplex_count: int = 16,
    precompute_resolution: int = 1008,
    interpol_size: list = None,
):
    """Memory encoder — resolution-aware version of _create_multiplex_maskmem_backbone."""
    if interpol_size is None:
        interpol_size = [1152, 1152]

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=interpol_size,
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _build_multiplex_transformer(
    feat_size: int = 72,
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    """Memory attention transformer — resolution-aware version of _create_multiplex_transformer."""
    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )


def _build_multiplex_tri_backbone(
    image_size: int = 1008,
    compile_mode=None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    """TriHead vision backbone — resolution-aware version of _create_multiplex_tri_backbone."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=image_size,
    )
    vit_backbone = ViT(
        img_size=image_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    tri_neck = Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
    return tri_neck
