# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import copy
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor

import dinov3.distributed as distributed
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.loss import KoLeoLoss
from dinov3.models import build_model, build_model_from_cfg
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp

logger = logging.getLogger("dinov3")


class FeatureDistillationMetaArch(nn.Module):
    """modified by zhoujiwen: head-free feature distillation from DINOv3 ViT-L to Swin."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        student_backbone, ema_backbone, embed_dim = build_model_from_cfg(cfg)
        teacher_args = copy.deepcopy(cfg.student)
        teacher_args.arch = cfg.feature_distill.teacher_arch
        teacher_args.patch_size = cfg.feature_distill.teacher_patch_size
        teacher_args.qkv_bias = True
        teacher_args.ffn_layer = "mlp"
        teacher_args.ffn_ratio = 4.0
        teacher_args.norm_layer = "layernormbf16"
        teacher_args.n_storage_tokens = 4
        teacher_args.mask_k_bias = True
        teacher_args.pos_embed_rope_dtype = "fp32"
        teacher_args.pos_embed_rope_rescale_coords = 2
        teacher_args.untie_cls_and_patch_norms = False
        teacher_args.untie_global_and_local_cls_norm = False
        teacher_backbone, teacher_dim = build_model(teacher_args, only_teacher=True, img_size=cfg.crops.global_crops_size, device="meta")
        self.student = nn.ModuleDict({"backbone": student_backbone})
        self.model_ema = nn.ModuleDict({"backbone": ema_backbone})
        self.teacher = nn.ModuleDict({"backbone": teacher_backbone})
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.embed_dim = embed_dim
        self.teacher_dim = teacher_dim
        self.has_gram_teacher = False
        self.koleo_loss = KoLeoLoss()
        self.ema_params_lists = None

    def init_weights(self) -> None:
        self.student.backbone.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        self._load_teacher_checkpoint()
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def _load_teacher_checkpoint(self) -> None:
        # modified by zhoujiwen: load a plain backbone checkpoint into the FSDP-sharded teacher.
        state = torch.load(self.cfg.feature_distill.teacher_checkpoint, map_location="cpu")
        state = state.get("teacher", state.get("model", state.get("state_dict", state))) if isinstance(state, dict) else state
        state = {f"backbone.{k.replace('module.', '').replace('backbone.', '')}": v for k, v in state.items()}
        mesh = DeviceMesh.from_group(distributed.get_default_process_group(), "cuda") if dist.is_initialized() else init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("dp",))
        state = {k: v if any(s in k for s in ("rope_embed.periods", "qkv.bias_mask")) else distribute_tensor(v, mesh, src_data_rank=None) for k, v in state.items()}
        self.teacher.load_state_dict(state, strict=False)

    def prepare_for_distributed_training(self) -> None:
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=[self.model_ema, self.teacher],
            cfg=self.cfg,
            trained_model_process_group=distributed.get_process_subgroup(),
            inference_only_models_process_groups=[distributed.get_process_subgroup(), distributed.get_default_process_group()],
        )

    def build_data_augmentation_dino(self, cfg):
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=None,
            gram_teacher_no_distortions=False,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def _feature_loss(self, student_feats, teacher_feats):
        return (1 - F.cosine_similarity(F.normalize(student_feats.float(), dim=-1), F.normalize(teacher_feats.float(), dim=-1), dim=-1)).mean()

    def _upsample_student_patches(self, patches, target_tokens):
        b, n, c = patches.shape
        src = int(n**0.5)
        dst = int(target_tokens**0.5)
        return F.interpolate(patches.transpose(1, 2).reshape(b, c, src, src), size=(dst, dst), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def _forward_loss(self, data):
        n_global = 2
        n_local = self.cfg.crops.local_crops_number
        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        b = global_crops.shape[0] // n_global
        with torch.no_grad():
            t_global = self.teacher.backbone(global_crops, is_training=True)
            t_local = self.teacher.backbone(local_crops, is_training=True) if n_local else None
        s_global = self.student.backbone(global_crops, masks=masks, is_training=True)
        s_local = self.student.backbone(local_crops, masks=None, is_training=True) if n_local else None
        s_patch = self._upsample_student_patches(s_global["x_norm_patchtokens"], t_global["x_norm_patchtokens"].shape[1])
        mask_grid = F.interpolate(masks.float().reshape(n_global * b, 1, 2, 2), size=(4, 4), mode="nearest").flatten(1).bool()
        cls_loss = self._feature_loss(s_global["x_norm_clstoken"], t_global["x_norm_clstoken"])
        if n_local:
            cls_loss = 0.5 * (cls_loss + self._feature_loss(s_local["x_norm_clstoken"], t_local["x_norm_clstoken"]))
        patch_loss = self._feature_loss(s_patch, t_global["x_norm_patchtokens"])
        masked_patch_loss = self._feature_loss(s_patch[mask_grid], t_global["x_norm_patchtokens"][mask_grid]) if mask_grid.any() else patch_loss.new_zeros(())
        koleo_loss = sum(self.koleo_loss(x) for x in s_global["x_norm_clstoken"].unflatten(0, (n_global, b))) / n_global
        total = (
            self.cfg.feature_distill.cls_loss_weight * cls_loss
            + self.cfg.feature_distill.patch_loss_weight * patch_loss
            + self.cfg.feature_distill.masked_patch_loss_weight * masked_patch_loss
            + self.cfg.dino.koleo_loss_weight * n_global * koleo_loss
        )
        return total, {"cls_feature_loss": cls_loss, "patch_feature_loss": patch_loss, "masked_patch_feature_loss": masked_patch_loss, "koleo_loss": koleo_loss}

    def forward_backward(self, data, *, teacher_temp=None, iteration=0, **ignored_kwargs) -> tuple[Tensor, dict[str, float | Tensor]]:
        del teacher_temp, iteration, ignored_kwargs
        loss, metrics = self._forward_loss(data)
        loss.backward()
        return loss, metrics

    @torch.no_grad()
    def validation_step(self, data):
        return self._forward_loss(data)

    def update_ema(self, m):
        if self.ema_params_lists is None:
            self.ema_params_lists = ([p for m_ in self.student.values() for p in m_.parameters()], [p for m_ in self.model_ema.values() for p in m_.parameters()])
        student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def get_params_groups(self):
        groups = []
        for m in self.student.values():
            groups += get_params_groups_with_decay_fsdp(m, self.cfg.optim.layerwise_decay, self.cfg.optim.patch_embed_lr_mult, self.cfg.optim.dino_head_wd_multiplier)
        return fuse_params_groups(groups) if self.cfg.optim.multi_tensor_optim else groups

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self
