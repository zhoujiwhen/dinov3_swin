# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
from pathlib import Path
from typing import Callable, Optional

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class FlatJPGDataset(ExtendedVisionDataset):
    Split = str

    def __init__(
        self,
        *,
        split: str,
        root: str,
        seed: int = 42,
        train_ratio: float = 0.9,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # modified by zhoujiwen: flat recursive JPG/PNG dataset for feature distillation.
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        paths = sorted(paths)
        rng = random.Random(seed)
        rng.shuffle(paths)
        split_at = int(len(paths) * train_ratio)
        self._paths = paths[:split_at] if split.upper() == "TRAIN" else paths[split_at:]

    def get_image_data(self, index: int) -> bytes:
        return self._paths[index].read_bytes()

    def get_target(self, index: int):
        return 0

    def __len__(self) -> int:
        return len(self._paths)
