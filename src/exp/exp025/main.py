import datetime
import os
from typing import Any, Dict, Generator, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import transforms as T

from src.exp.exp025.config import InputPath, ModelConfig, OutputPath
from src.utils import DefaultLogger, Jbl, Logger, fix_seed

sns.set_style("whitegrid")


def validate_config(model_config: ModelConfig) -> None:
    def _validate_run_name(model_config: ModelConfig) -> None:
        # If you want to remove models, run `rm output/model/*expXXX*` in the root dir.
        past_sessions = [
            x.split("_")[0]
            for x in os.listdir(OutputPath.model)
            if x.endswith("_0.pth")
        ]
        assert model_config.basic.run_name not in past_sessions

    def _validate_device(model_config: ModelConfig) -> None:
        assert model_config.basic.device == "cuda"

    _validate_run_name(model_config)
    _validate_device(model_config)


def load_npz(path: str) -> np.array:
    x = np.load(path)["arr_0"]
    return x


def generate_kf(cfg: ModelConfig) -> Generator:
    if cfg.kfold.method == "kf":
        kf = KFold(
            n_splits=cfg.kfold.number,
            shuffle=cfg.kfold.shuffle,
            random_state=cfg.basic.seed,
        )
    elif cfg.kfold.method == "skf":
        kf = StratifiedKFold(
            n_splits=cfg.kfold.number,
            shuffle=cfg.kfold.shuffle,
            random_state=cfg.basic.seed,
        )
    elif cfg.kfold.method == "gkf":
        kf = GroupKFold(n_splits=cfg.kfold.number)
    elif cfg.kfold.method == "sgkf":
        raise ValueError("kfold method sgkf is not implemented")
        # kf = StratifiedGroupKFold(
        #     n_splits=cfg.kfold.number, random_state=cfg.basic.seed
        # )
    elif cfg.kfold.method == "tskf":
        kf = TimeSeriesSplit(n_splits=cfg.kfold.number)
    else:
        raise ValueError(f"{cfg.kfold.method} is not supported")
    return kf


class ProbSpaceDataset(data.Dataset):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def __init__(
        self, images: np.array, labels: Optional[np.array] = None, is_train: bool = True
    ) -> None:
        """images.shape: (b, h, w, c), labels: (b,)"""
        assert (is_train and labels is not None) or (not is_train and labels is None)
        self.is_train = is_train
        self.images = images
        self.labels = labels

        size = (ModelConfig.params.image_size, ModelConfig.params.image_size)
        additional_items = (
            [
                T.ToPILImage(),
                T.Resize(size),
            ]
            if not is_train
            else [
                T.ToPILImage(),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.3,
                    contrast=0.5,
                    saturation=[0.8, 1.3],
                    hue=[-0.05, 0.05],
                ),
                T.RandomResizedCrop(size),
            ]
        )
        self.transformer = T.Compose(
            [
                *additional_items,
                T.ToTensor(),
                T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD),
            ]
        )

    def __getitem__(self, index) -> Dict[str, Any]:
        image = self.images[index]
        image = self.transformer(image)
        if self.is_train:
            label = self.labels[index]
        else:
            label = -1
        return {"image": image, "label": label}

    def __len__(self) -> int:
        return len(self.images)


class ProbSpaceModel(nn.Module):
    def __init__(self, model_config: ModelConfig, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=model_config.params.pretrained,
            num_classes=model_config.params.target_size,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x


def build_model(model_config: ModelConfig, model_name: str):
    model = ProbSpaceModel(model_config, model_name)
    model.to(model_config.basic.device)
    return model


class BaseRunner:
    def __init__(self, cfg: ModelConfig, logger: Optional[Logger] = None):
        self.cfg = cfg
        self.params = cfg.params
        if logger is not None:
            self.logger = logger
        else:
            logger = DefaultLogger()
            self.logger = logger
        self.logger.info(self.cfg)

    def _get_scheduler(
        self, optimizer: Union[optim.Adam]
    ) -> Union[lr_scheduler.ReduceLROnPlateau]:
        if self.params.scheduler.name == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.params.scheduler.mode,
                factor=self.params.scheduler.factor,
                patience=self.params.scheduler.patience,
                verbose=self.params.scheduler.verbose,
                eps=self.params.scheduler.eps,
            )
        else:
            raise ValueError(f"{self.params.scheduler.name} is not supported")
        return scheduler

    def _step_scheduler(
        self,
        scheduler: Union[
            lr_scheduler.ReduceLROnPlateau,
        ],
        avg_val_loss,
    ) -> Union[lr_scheduler.ReduceLROnPlateau]:
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            raise ValueError(f"{self.params.shceduler.name} is not supported")
        return scheduler

    def _evaluate(
        self, y_true: np.array, y_pred: np.array, verbose: bool = False
    ) -> float:
        score = metrics.accuracy_score(y_true, y_pred)
        if verbose:
            self.logger.info(f"Score: {score:<.5f}")
        return score


class StackingRunner(BaseRunner):
    def _train(self, train: pd.DataFrame, n_fold: int):
        self.logger.info(f"fold: {n_fold}")

        val_idx = train[train["fold"] == n_fold].index.tolist()
        valid_folds = train.loc[val_idx].reset_index(drop=True)

        preds: List[np.array] = []
        for run_name in self.cfg.stacked_features:
            model_dict = torch.load(f"{OutputPath.model}/{run_name}_{n_fold}.pth")
            pred = model_dict["preds"]
            if run_name == "exp018":
                preds_evaluate_length = model_dict["preds_evaluate_length"]
                pred = pred[:preds_evaluate_length]
            preds.append(pred)
        preds = np.mean(preds, axis=0)
        torch.save(
            {
                "preds": preds,
                "config": self.cfg,
            },
            f"{OutputPath.model}/{self.cfg.basic.run_name}_{n_fold}.pth",
        )
        valid_folds = valid_folds.assign(preds=np.argmax(preds, axis=1))
        return valid_folds

    def run_cv(self, train: pd.DataFrame) -> None:
        self.logger.info(f"debug mode: {self.cfg.basic.is_debug}")
        self.logger.info(f"start time: {datetime.datetime.now()}")
        oof_df = pd.DataFrame()
        for n_fold in range(self.cfg.kfold.number):
            _oof_df = self._train(train, n_fold)
            self.logger.info(f"========== fold: {n_fold} result ==========")
            score = self._evaluate(_oof_df["target"], _oof_df["preds"], verbose=True)
            if hasattr(self.logger, "result"):
                self.logger.result(f"Fold {n_fold} Score: {score:<.5f}")
            oof_df = pd.concat([oof_df, _oof_df])
        self.logger.info("========== CV ==========")
        score = self._evaluate(oof_df["target"], oof_df["preds"], verbose=True)
        if hasattr(self.logger, "result"):
            self.logger.result(f"CV Score: {score:<.5f}")
        Jbl.save(oof_df, f"{OutputPath.model}/oof_df_{self.cfg.basic.run_name}.jbl")


class StackingInferenceRunner(BaseRunner):
    def _test(self, test: pd.DataFrame, n_fold: int):
        self.logger.info(f"fold: {n_fold}")

        preds: List[np.array] = []
        for run_name in self.cfg.stacked_features:
            preds_array = Jbl.load(f"{OutputPath.model}/preds_test_{run_name}.jbl")
            preds_mean = np.mean(preds_array, axis=0)
            preds.append(preds_mean)
        preds = np.mean(preds, axis=0)
        return preds

    def _submit(self, preds: np.array) -> None:
        test_images = load_npz(InputPath.test_images)
        df_sub = pd.DataFrame({"id": list(range(len(test_images)))})
        df_sub = df_sub.assign(y=preds)
        self.logger.info(df_sub.head())
        df_sub = df_sub.astype(int)
        path = f"{OutputPath.submission}/submission_{self.cfg.basic.run_name}.csv"
        df_sub.to_csv(path, index=False)
        self.logger.info("submission.csv created")

    def run_cv(
        self,
        test: Optional[pd.DataFrame] = None,
    ) -> None:
        # oof_df = pd.DataFrame()
        preds: List[np.array] = []
        for n_fold in range(self.cfg.kfold.number):
            preds_fold = self._test(test, n_fold)
            preds.append(preds_fold)
        Jbl.save(preds, f"{OutputPath.model}/preds_test_{self.cfg.basic.run_name}.jbl")

        preds_mean = np.mean(preds, axis=0)
        assert preds_mean.shape == (497, 13)
        preds_mean = preds_mean.argmax(axis=1)
        assert preds_mean.shape == (497,)
        self._submit(preds_mean)


def visualize_prediction(model_config: ModelConfig, logger: Logger) -> None:
    logger.info(
        Jbl.load(f"{OutputPath.model}/oof_df_{model_config.basic.run_name}.jbl").head()
    )
    sns.countplot(
        x=Jbl.load(f"{OutputPath.model}/oof_df_{model_config.basic.run_name}.jbl")[
            "preds"
        ]
    )
    plt.savefig(f"{OutputPath.model}/pred_countplot_{model_config.basic.run_name}.png")


def main():
    fix_seed()
    model_config = ModelConfig()
    validate_config(model_config)

    train_labels = load_npz(InputPath.train_labels)
    train = pd.DataFrame({"target": train_labels})

    kf = generate_kf(model_config)
    kf_generator = kf.split(train, train["target"])
    for fold_i, (tr_idx, val_idx) in enumerate(kf_generator):
        train.loc[val_idx, "fold"] = fold_i
    train = train.assign(fold=train["fold"].astype(int))

    run_name = model_config.basic.run_name
    logger = Logger(
        f"{OutputPath.logs}/{run_name}/general.log",
        f"{OutputPath.logs}/{run_name}/result.log",
        run_name,
    )
    StackingRunner(model_config, logger).run_cv(train)
    StackingInferenceRunner(model_config, logger).run_cv()

    visualize_prediction(model_config, logger)
