"""
base: exp028
change: stacking: exp028, exp027, exp026, exp025, exp024, exp023 (average)
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
    run_name: str = "exp029"
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
    target_size: int = 13


@dataclass
class ModelConfig:
    basic: Basic = Basic()
    kfold: Kfold = Kfold()
    params: Params = Params()
    stacked_features: List[str] = field(
        default_factory=lambda: [
            "exp028",
            "exp027",
            "exp026",
            "exp025",
            "exp024",
            "exp023",
        ]
    )
