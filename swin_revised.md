# SwinTransformer 修改部分
## 1.SwinTransformer前向部分
### 1.1 window并行修改
复用ibot当中的swintransformer 新增SwinTransformerBlock当中的_forward_list函数

```python
def _forward_list(self, x_list):
        shortcut = []
        x_windows = []
        attn_mask = []
        spatial_shape = []
        ori_shape = []
        pad = []        
        x = []
        for x_ in x_list:
            B, L, C = x_.shape
            H = int(sqrt(L))
            W = H
            ori_shape.append((B, H, W, C))
            shortcut_ = x_
            shortcut.append(shortcut_)
            x_ = self.norm1(x_)
            x_ = x_.view(B, H, W, C)
            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad.append((pad_l, pad_r, pad_t, pad_b))
            x_ = F.pad(x_, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x_.shape
            spatial_shape.append((Hp, Wp))
            # cyclic shift
            if self.shift_size > 0:
                shifted_x_ = torch.roll(x_, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                if H is self.attn_mask_dict.keys():
                    attn_mask_ = self.attn_mask_dict[H]
                else:
                    self.attn_mask_dict[H] = self.create_attn_mask(H, W).to(x_.device).to(x_.dtype)
                    attn_mask_ = self.attn_mask_dict[H]
                attn_mask_ = attn_mask_.unsqueeze(0).repeat(B, 1, 1, 1)
                attn_mask_ = attn_mask_.reshape(-1, self.window_size*self.window_size, self.window_size*self.window_size)
            else:
                shifted_x_ = x_
                attn_mask_ = None
            attn_mask.append(attn_mask_)
            # nW1*B1, window_size, window_size, C  nW2*B2, window_size, window_size, C
            # partition windows              
            x_windows_ = window_partition(shifted_x_, self.window_size)  # nW*B, window_size, window_size, C
            x_windows_ = x_windows_.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            x_windows.append(x_windows_)
            # W-MSA/SW-MSA
        b_list = [x_windows_.shape[0] for x_windows_ in x_windows]
        x_windows = torch.cat(x_windows, dim=0)  # nW1*B1 + nW2*B2, window_size*window_size, C
        attn_mask = torch.cat(attn_mask, dim=0) if attn_mask[0] is not None else None
        attn_windows, attn = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C

        attn_windows = torch.split(attn_windows, b_list, dim=0)
        for attn_windows_, spatial_shape_, pad_, ori_shape_, shortcut_ in zip(attn_windows, spatial_shape, pad, ori_shape, shortcut):
            Hp, Wp = spatial_shape_
            B, H, W, C = ori_shape_
            # merge windows
            attn_windows_ = attn_windows_.view(b_list[0], self.window_size, self.window_size, -1)
        
            shifted_x_ = window_reverse(attn_windows_, self.window_size, Hp, Wp)  # B H' W' C
            
            # reverse cyclic shift
            if self.shift_size > 0:
                x_ = torch.roll(shifted_x_, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x_ = shifted_x_
            pad_l, pad_r, pad_t, pad_b = pad_
            if pad_r > 0 or pad_b > 0:
                x_ = x_[:, :H, :W, :].contiguous()

            x_ = x_.view(B, H * W, C)

            # FFN
            x_ = shortcut_ + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            x.append(x_)
        return x, attn
```
其中关键的部分是将global和local的所有特征做window切分 切成不同的window特征 对应代码为
```python
# partition windows              
x_windows_ = window_partition(shifted_x_, self.window_size)  # nW*B, window_size, window_size, C
```

其中 global input的window特征为 B_global*nW_global，window_size, window_size, C

local input的window特征为 B_local*nW_local，window_size, window_size, C

前向传播时在第一个维度做cat 其特征维度为 （B_global * nW_global + B_local*nW_local，window_size, window_size, C）

 经过window attention之后再torch.split 为global特征（B_global\*nW_global，window_size, window_size, C）和  local特征 （B_local*nW_local，window_size, window_size, C）
```python
b_list = [x_windows_.shape[0] for x_windows_ in x_windows]
x_windows = torch.cat(x_windows, dim=0)  # nW1*B1 + nW2*B2, window_size*window_size, C
attn_mask = torch.cat(attn_mask, dim=0) if attn_mask[0] is not None else None
attn_windows, attn = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C
```
attention mask也做了修改 加了如下代码：
```python
attn_mask_ = attn_mask_.unsqueeze(0).repeat(B, 1, 1, 1)
attn_mask_ = attn_mask_.reshape(-1, self.window_size*self.window_size, self.window_size*self.window_size)
```
因为原始代码中attn_mask_的维度是nW, window_size\*window_size, window_size\*window_size
原始代码在batch维度做的unsqueeze
现在由于需要做window attention的并行，而global和local 的window attention mask是不一样的（nW_global！=nW_local），因此现在需要直接对齐window特征维度，而不是直接unsqueeze，扩展attn_mask为B\*nW, window_size\*window_size, window_size\*window_size。
### 1.2 前向传播函数修改
在SwinTransformer的前向函数中分为_forward_list和_forward 分别处理单个tensor和多个tensor 根据输入的type决定
```pyhton
def forward(self, x):
    if type(x) == list:
        return self._forward_list(x)
    else:
        return self._forward(x)
```
## 2.模型与dataloader初始化相关修改

### 2.1 YAML相关修改
```yaml
MODEL:
  META_ARCHITECTURE: SSLMetaArch

# 使用 ViT-Small 进行快速训练
student:
  arch: swin_small
  patch_size: 4
  drop_path_rate: 0.1
  qkv_bias: true
  # proj_bias: true
  # ffn_bias: true
  norm_layer: layernorm
```
student初始化对应transformer中的swin_small函数
```python
@register_model
def swin_small(window_size=7, **kwargs):
    # model = SwinTransformer(
    #     window_size=window_size, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
    #     mlp_ratio=4, qkv_bias=True, drop_path_rate=kwargs.pop('drop_path_rate', 0.2), **kwargs)
    # return model
    model = SwinTransformer(
        window_size=window_size, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        mlp_ratio=4, drop_path_rate=kwargs.pop('drop_path_rate', 0.2), **kwargs)
    return model
```
此外，在ssl_meta_arch.py中修改了 DinoHead的in_dim(swintransformer中的embed_dim为模型的第一层特征维度，最后一层特征变为了第一层特征的8倍)
```python
dino_head_class = partial(
    DINOHead,
    in_dim=embed_dim if type(student_backbone) != SwinTransformer else embed_dim*8,
    out_dim=cfg.dino.head_n_prototypes,
    hidden_dim=cfg.dino.head_hidden_dim,
    bottleneck_dim=cfg.dino.head_bottleneck_dim,
    nlayers=cfg.dino.head_nlayers,
)
```

### 2.2 FSDP&梯度检查点与模型编译相关修改
ac_compile里修改相关函数 swin transformer里面用的是layer 而不是blocks做的前向传播这里新增了fsdp_transformer_swin compile_transformer_swin
```python
def fsdp_transformer_swin(fsdp_config: Dict[str, Any], model: nn.Module):
    # Backbone - FSDP every block
    blocks = model.layers
    assert isinstance(blocks, nn.ModuleList)
    for block_id, block in enumerate(blocks):
        block_reshard: int | bool = True
        blocks[block_id] = fully_shard(block, **fsdp_config, reshard_after_forward=block_reshard)
    prev_block: FSDPState
    next_block: FSDPState
    for prev_block, next_block in zip(blocks, blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
        next_block.set_modules_to_backward_prefetch([prev_block])
    fully_shard(model, **fsdp_config, reshard_after_forward=True)
    register_fsdp_forward_method(model, "get_intermediate_layers")

def compile_transformer_swin(cfg, model: nn.Module):
    assert isinstance(model.layers, nn.ModuleList)
    for block_id, block in enumerate(model.layers):
        model.layers[block_id] = wrap_compile_block(block, cfg.train.cudagraphs, is_backbone_block=True)
```
### 2.3 适配Swin的MaskGenerator修改
在/mnt/qh2-nas3/00-model/00-wrs/dinov3_workspace/code_revised/dinov3-swin/dinov3/train/train.py
中修改build_data_loader_from_cfg函数
因为mask是mask最后特征的，而swintransformer是下采样了三次，空间尺度为原来的1/8 所以如果是swintransformer的话  patch_size是原来的8倍
```python
def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    swin_flag = 'SwinTransformer' in type(model.student['backbone']).__name__
    swin_patch_size = patch_size * 8
    n_tokens = (img_size // patch_size) ** 2 if not swin_flag else (img_size // swin_patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size) if not swin_flag else (img_size // swin_patch_size, img_size // swin_patch_size),
        # max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        max_num_patches=0.5*n_tokens,
    )
```
 