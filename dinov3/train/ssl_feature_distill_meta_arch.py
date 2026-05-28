# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import copy
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor

import dinov3.distributed as distributed
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.loss import GramLoss, KoLeoLoss
from dinov3.models import build_model, build_model_from_cfg
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp

logger = logging.getLogger("dinov3")


class SSLFeatureDistillationMetaArch(nn.Module):
    """modified by zhoujiwen: DINO/iBOT/KoLeo/Gram structure with feature-space cosine DINO/iBOT."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        student_backbone, ema_backbone, embed_dim = build_model_from_cfg(cfg)
        teacher_args = copy.deepcopy(cfg.student)
        teacher_args.arch, teacher_args.patch_size = cfg.feature_distill.teacher_arch, cfg.feature_distill.teacher_patch_size
        teacher_args.qkv_bias, teacher_args.ffn_layer, teacher_args.ffn_ratio, teacher_args.norm_layer = True, "mlp", 4.0, "layernormbf16"
        teacher_args.n_storage_tokens, teacher_args.mask_k_bias, teacher_args.pos_embed_rope_dtype, teacher_args.pos_embed_rope_rescale_coords = 4, True, "fp32", 2
        teacher_args.untie_cls_and_patch_norms = teacher_args.untie_global_and_local_cls_norm = False
        teacher_backbone, teacher_dim = build_model(teacher_args, only_teacher=True, img_size=cfg.crops.global_crops_size, device="meta")
        self.student, self.model_ema, self.teacher = nn.ModuleDict({"backbone": student_backbone}), nn.ModuleDict({"backbone": ema_backbone}), nn.ModuleDict({"backbone": teacher_backbone})
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.embed_dim, self.teacher_dim, self.has_gram_teacher, self.ema_params_lists = embed_dim, teacher_dim, False, None
        self.koleo_loss = KoLeoLoss()
        self.gram_use_loss = cfg.gram.use_loss
        self.gram_loss = GramLoss(apply_norm=cfg.gram.normalized, remove_only_teacher_neg=cfg.gram.remove_only_teacher_neg, remove_neg=cfg.gram.remove_neg) if self.gram_use_loss else None
        self.gram_loss_weight, self.gram_img_level, self.gram_tokens_used, self.gram_compute_stats = cfg.gram.loss_weight, cfg.gram.img_level, cfg.gram.tokens_used, cfg.gram.compute_stats

    def init_weights(self) -> None:
        self.student.backbone.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        self._load_teacher_checkpoint()
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def _load_teacher_checkpoint(self) -> None:
        # modified by zhoujiwen: keep the original checkpoint path, but load only the no-head teacher backbone.
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
        return DataAugmentationDINO(cfg.crops.global_crops_scale, cfg.crops.local_crops_scale, cfg.crops.local_crops_number, global_crops_size=cfg.crops.global_crops_size, local_crops_size=cfg.crops.local_crops_size, gram_teacher_crops_size=None, gram_teacher_no_distortions=False, local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops, share_color_jitter=cfg.crops.share_color_jitter, horizontal_flips=cfg.crops.horizontal_flips, mean=cfg.crops.rgb_mean, std=cfg.crops.rgb_std)

    def _feature_loss(self, student_feats, teacher_feats):
        return (1 - F.cosine_similarity(F.normalize(student_feats.float(), dim=-1), F.normalize(teacher_feats.float(), dim=-1), dim=-1)).mean()

    def _upsample_student_patches(self, patches, target_tokens):
        b, n, c = patches.shape
        src, dst = int(n**0.5), int(target_tokens**0.5)
        return F.interpolate(patches.transpose(1, 2).reshape(b, c, src, src), size=(dst, dst), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def _resize_masks(self, masks, target_tokens):
        src, dst = int(masks.shape[1] ** 0.5), int(target_tokens**0.5)
        return F.interpolate(masks.float().reshape(masks.shape[0], 1, src, src), size=(dst, dst), mode="nearest").flatten(1).bool()

    def _forward_loss(self, data):
        n_global, n_local = 2, self.cfg.crops.local_crops_number
        global_crops, local_crops, masks = data["collated_global_crops"].cuda(non_blocking=True), data["collated_local_crops"].cuda(non_blocking=True), data["collated_masks"].cuda(non_blocking=True)
        b = global_crops.shape[0] // n_global
        with torch.no_grad():
            t_global = self.teacher.backbone(global_crops, is_training=True)
        s_global = self.student.backbone(global_crops, masks=masks, is_training=True)
        s_local = self.student.backbone(local_crops, masks=None, is_training=True) if n_local else None
        t_cls_img = t_global["x_norm_clstoken"].unflatten(0, (n_global, b)).mean(0)
        dino_global_loss = self._feature_loss(s_global["x_norm_clstoken"], t_global["x_norm_clstoken"])
        dino_local_loss = self._feature_loss(s_local["x_norm_clstoken"], t_cls_img.repeat(n_local, 1)) if n_local else dino_global_loss.new_zeros(())
        dino_loss = self.cfg.dino.loss_weight * 0.5 * (dino_global_loss + dino_local_loss)
        s_patch = self._upsample_student_patches(s_global["x_norm_patchtokens"], t_global["x_norm_patchtokens"].shape[1])
        mask_grid = self._resize_masks(masks, t_global["x_norm_patchtokens"].shape[1])
        ibot_loss = self.cfg.ibot.loss_weight * (self._feature_loss(s_patch[mask_grid], t_global["x_norm_patchtokens"][mask_grid]) if mask_grid.any() else dino_loss.new_zeros(()))
        koleo_loss = self.cfg.dino.koleo_loss_weight * sum(self.koleo_loss(x) for x in s_global["x_norm_clstoken"].unflatten(0, (n_global, b))) / n_global
        gram_student, gram_teacher = (s_patch[mask_grid], t_global["x_norm_patchtokens"][mask_grid]) if self.gram_tokens_used == "masked" and mask_grid.any() else (s_patch[~mask_grid], t_global["x_norm_patchtokens"][~mask_grid]) if self.gram_tokens_used == "unmasked" and (~mask_grid).any() else (s_patch, t_global["x_norm_patchtokens"])
        gram_loss = self.gram_loss_weight * self.gram_loss(gram_student, gram_teacher, img_level=self.gram_img_level) if self.gram_use_loss else dino_loss.new_zeros(())
        total = dino_loss + ibot_loss + koleo_loss + gram_loss
        return total, {"dino_loss": dino_loss.detach(), "ibot_loss": ibot_loss.detach(), "koleo_loss": koleo_loss.detach(), "gram_loss": gram_loss.detach()}

    def forward_backward(self, data, *, teacher_temp=None, iteration=0, **ignored_kwargs) -> tuple[Tensor, dict[str, Tensor]]:
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
