"""
base: exp034
change: stacking: exp018, exp015, exp012, exp010 (2nd stage)
"""
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class InputPath:
    _prefix: str = "./input"
    train_images: str = f"{_prefix}/christ-train-imgs.npz"
    train_labels: str = f"{_prefix}/christ-train-labels.npz"
    test_images: str = f"{_prefix}/christ-test-imgs.npz"


@dataclass
class OutputPath:
    _prefix: str = "./output"
    logs: str = f"{_prefix}/logs"
    model: str = f"{_prefix}/model"
    submission: str = f"{_prefix}/submission"


@dataclass
class Basic:
    run_name: str = "exp035"
    is_debug: bool = False
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Kfold:
    number: int = 5
    method: str = "skf"
    shuffle: bool = True
    columns: List[str] = field(default_factory=lambda: ["target"])


@dataclass
class Adam:
    name: str = "Adam"
    lr: float = 1e-5
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class ReduceLROnPlateau:
    name: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 5
    verbose: bool = True
    eps: float = 1e-8


@dataclass
class Params:
    model_name: str = "resnext50_32x4d" if Basic.is_debug else "resnext101_32x8d"
    model_name_exp010: str = "resnest50d"
    model_name_exp012: str = "resnext50_32x4d"
    model_name_exp015: str = "resnext101_32x8d"
    model_name_exp018: str = "resnext101_32x8d"
    batch_size: int = 16 if Basic.is_debug else 8
    test_batch_size: int = 128 if Basic.is_debug else 64
    epochs: int = 3 if Basic.is_debug else 200
    image_size: int = 224 if Basic.is_debug else 448
    num_workers: int = 0
    target_size: int = 13
    # Union[Adam]
    optimizer: Adam = Adam()
    # Union[CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau]
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau()
    pretrained: bool = True
    num_aug: int = 1
    num_tta: int = 1
    early_stopping_rounds: int = 10
    is_psuedo_labeling: bool = False


@dataclass
class ModelConfig:
    basic: Basic = Basic()
    kfold: Kfold = Kfold()
    params: Params = Params()
    stacked_features: List[str] = field(
        default_factory=lambda: [
            "exp018",
            "exp015",
            "exp012",
            "exp010",
        ]
    )
