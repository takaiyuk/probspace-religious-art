"""
base: exp016
change: stacking: exp008, exp010, exp012, exp015 (average)
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
    run_name: str = "exp016"
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
    model_name_exp008: str = "resnet50"
    model_name_exp010: str = "resnest50d"
    model_name_exp012: str = "resnext50_32x4d"
    model_name_exp015: str = "resnext101_32x8d"
    batch_size: int = 8
    test_batch_size: int = 64
    epochs: int = 3 if Basic.is_debug else 200
    image_size: int = 448
    num_workers: int = 0
    target_size: int = 13
    # Union[Adam]
    optimizer: Adam = Adam()
    # Union[CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau]
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau()
    pretrained: bool = True
    num_aug: int = 10
    num_tta: int = 10
    early_stopping_rounds: int = 10


@dataclass
class ModelConfig:
    basic: Basic = Basic()
    kfold: Kfold = Kfold()
    params: Params = Params()
    stacked_features: List[str] = field(
        default_factory=lambda: ["exp008", "exp010", "exp012", "exp015"]
    )
