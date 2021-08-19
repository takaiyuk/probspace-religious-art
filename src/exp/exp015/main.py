import datetime
import os
import time
from collections import Counter, OrderedDict
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
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.exp.exp015.config import InputPath, ModelConfig, OutputPath
from src.utils import AverageMeter, DefaultLogger, Jbl, Logger, fix_seed

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
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.backbone = timm.create_model(
            model_config.params.model_name,
            pretrained=model_config.params.pretrained,
            num_classes=model_config.params.target_size,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x


def build_model(model_config: ModelConfig):
    model = ProbSpaceModel(model_config)
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


class TrainRunner(BaseRunner):
    def _train_epoch(self, train_loader, model, criterion, optimizer, scheduler, epoch):
        losses = AverageMeter()
        model.train()
        for _ in range(self.cfg.params.num_aug):
            for step, image_label_dict in enumerate(train_loader):
                images = image_label_dict.get("image").to(self.cfg.basic.device)
                labels = image_label_dict.get("label").to(self.cfg.basic.device)
                batch_size = labels.size(0)

                y_preds = model(images)
                loss = criterion(y_preds, labels)
                losses.update(loss.item(), batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return losses.avg

    def _valid_epoch(self, valid_loader, model, criterion):
        losses = AverageMeter()
        model.eval()
        preds = []
        for _, image_label_dict in enumerate(valid_loader):
            images = image_label_dict.get("image").to(self.cfg.basic.device)
            labels = image_label_dict.get("label").to(self.cfg.basic.device)
            batch_size = labels.size(0)

            with torch.no_grad():
                y_preds = model(images)
            loss = criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)
            preds.append(y_preds.to("cpu").numpy())
        predictions = np.concatenate(preds).reshape(-1, self.params.target_size)
        return losses.avg, predictions

    def _train(self, train: pd.DataFrame, n_fold: int) -> pd.DataFrame:
        self.logger.info(f"fold: {n_fold}")
        train_images = load_npz(InputPath.train_images)
        train_labels = load_npz(InputPath.train_labels)

        is_tta_mode = self.params.num_tta > 0
        num_times_tta = 1 if not is_tta_mode else self.params.num_tta

        trn_idx = train[train["fold"] != n_fold].index.tolist()
        val_idx = train[train["fold"] == n_fold].index.tolist()
        train_images_folds = train_images[trn_idx]
        valid_images_folds = train_images[val_idx]
        train_labels_folds = train_labels[trn_idx]
        valid_labels_folds = train_labels[val_idx]
        # train_folds = train.loc[trn_idx].reset_index(drop=True)
        valid_folds = train.loc[val_idx].reset_index(drop=True)
        train_dataset = ProbSpaceDataset(
            train_images_folds,
            train_labels_folds,
            is_train=True,
            #             transform=get_transforms(self.params, data="train"),
        )
        valid_dataset = ProbSpaceDataset(
            valid_images_folds,
            valid_labels_folds,
            is_train=is_tta_mode,
            #             transform=get_transforms(self.params, data="valid"),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # 少数クラスほど重みをつける
        weights = 1 / np.array(
            [Counter(train_labels_folds)[i] for i in range(self.params.target_size)]
        )
        weights = weights / np.sum(weights)
        assert np.all(weights != 0)
        weights = torch.tensor(weights).float().to(self.cfg.basic.device)

        model = build_model(model_config=self.cfg)
        model.to(self.cfg.basic.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.params.optimizer.lr,
            weight_decay=self.params.optimizer.weight_decay,
            amsgrad=self.params.optimizer.amsgrad,
        )
        scheduler = self._get_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_model = None
        best_preds = None
        best_score = 0
        scores: List[float] = []
        num_not_improved = 0
        for epoch in range(self.params.epochs):
            start_time = time.time()

            avg_loss = self._train_epoch(
                train_loader, model, criterion, optimizer, scheduler, epoch
            )
            avg_val_loss_list: List[float] = []
            preds_array = np.zeros(
                (num_times_tta, len(val_idx), self.params.target_size)
            )
            for i in range(num_times_tta):
                avg_val_loss, preds = self._valid_epoch(valid_loader, model, criterion)
                avg_val_loss_list.append(avg_val_loss)
                preds_array[i] = preds
            avg_val_loss = np.mean(avg_val_loss_list)
            scheduler = self._step_scheduler(scheduler, avg_val_loss)

            preds = preds_array.mean(axis=0)
            preds_ = np.argmax(preds, axis=1)
            valid_labels = valid_folds["target"].values
            score = self._evaluate(valid_labels, preds_)
            scores.append(score)
            elapsed = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )
            self.logger.info(f"Epoch {epoch+1} - Accuracy: {score}")
            if score > best_score:
                best_model = model
                best_preds = preds
                best_score = score
                num_not_improved = 0
            else:
                num_not_improved += 1
            self.logger.info(f"Epoch {epoch+1} - Best Score: {best_score:.4f}")
            if (
                self.params.early_stopping_rounds > 0
                and self.params.early_stopping_rounds == num_not_improved
            ):
                self.logger.info(
                    f"Early stopping break: not improved {num_not_improved} times in a row"
                )
                break

        torch.save(
            {
                "model": best_model.state_dict(),
                "preds": best_preds,
                "best_score": best_score,
                "scores": scores,
                "config": self.cfg,
            },
            f"{OutputPath.model}/{self.cfg.basic.run_name}_{n_fold}.pth",
        )
        check_point: Dict[str, Union[OrderedDict, torch.Tensor]] = torch.load(
            f"{OutputPath.model}/{self.cfg.basic.run_name}_{n_fold}.pth"
        )
        valid_folds["preds"] = np.argmax(check_point["preds"], axis=1)
        return valid_folds

    def run_cv(self, train: pd.DataFrame) -> None:
        self.logger.info(f"Runner: {self.__class__.__name__}")
        self.logger.info(f"debug mode: {self.cfg.basic.is_debug}")
        self.logger.info(f"start time: {datetime.datetime.now()}")
        oof_df = pd.DataFrame()
        for n_fold in range(self.cfg.kfold.number):
            start_time = time.time()
            _oof_df = self._train(train, n_fold)
            elapsed = time.time() - start_time
            self.logger.info(f"========== fold: {n_fold} result ==========")
            self.logger.info(f"fold{n_fold} time: {elapsed/60:.0f}min.")
            score = self._evaluate(_oof_df["target"], _oof_df["preds"], verbose=True)
            if hasattr(self.logger, "result"):
                self.logger.result(f"Fold {n_fold} Score: {score:<.5f}")
            oof_df = pd.concat([oof_df, _oof_df])
        self.logger.info("========== CV ==========")
        score = self._evaluate(oof_df["target"], oof_df["preds"], verbose=True)
        if hasattr(self.logger, "result"):
            self.logger.result(f"CV Score: {score:<.5f}")
        Jbl.save(oof_df, f"{OutputPath.model}/oof_df_{self.cfg.basic.run_name}.jbl")


class InferenceRunner(BaseRunner):
    def _test_epoch(self, test_loader, model):
        model.eval()
        preds = []
        for step, image_label_dict in enumerate(test_loader):
            images = image_label_dict.get("image").to(self.cfg.basic.device)
            with torch.no_grad():
                y_preds = model(images)
            preds.append(y_preds.to("cpu").numpy())
        predictions = np.concatenate(preds).reshape(-1, self.params.target_size)
        return predictions

    def _test(self, test: pd.DataFrame, n_fold: int):
        self.logger.info(f"fold: {n_fold}")
        test_images = load_npz(InputPath.test_images)

        is_tta_mode = self.params.num_tta > 0
        num_times_tta = 1 if not is_tta_mode else self.params.num_tta

        test_dataset = ProbSpaceDataset(
            test_images,
            is_train=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.test_batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = build_model(model_config=self.cfg)
        model_state = torch.load(
            f"{OutputPath.model}/{self.cfg.basic.run_name}_{n_fold}.pth"
        )["model"]
        model.load_state_dict(model_state)
        model.to(self.cfg.basic.device)
        # preds = self._test_epoch(test_loader, model)
        preds_array = np.zeros(
            (num_times_tta, len(test_images), self.params.target_size)
        )
        for i in range(num_times_tta):
            _preds = self._test_epoch(test_loader, model)
            preds_array[i] = _preds
        preds = preds_array.mean(axis=0)
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

    def run_cv(self, test: pd.DataFrame = None) -> None:
        self.logger.info(f"Runner: {self.__class__.__name__}")
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
    TrainRunner(model_config, logger).run_cv(train)
    InferenceRunner(model_config, logger).run_cv()

    visualize_prediction(model_config, logger)
