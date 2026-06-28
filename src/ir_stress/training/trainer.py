"""Contrast-Phys+ training loop with MLflow tracking.

Training procedure adapted from the Contrast-Phys+ reference implementation:
  https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B
  (upstream: contrast-phys+/train.py)

Original work: Sun & Li, TPAMI 2024.
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ir_stress.config import TrainConfig, resolve_h5_dir
from ir_stress.dataset.dataset import H5ClipDataset
from ir_stress.device import device_label, resolve_device, supports_amp
from ir_stress.dataset.splits import resolve_h5_splits, save_split_metadata
from ir_stress.models.model import RppgModel, build_model
from ir_stress.tracking import ensure_mlflow
from ir_stress.training.ipr import IrrelevantPowerRatio
from ir_stress.training.loss import ContrastLoss


def _forward_model(model: RppgModel, imgs: torch.Tensor, micro_batch: bool) -> torch.Tensor:
    """Forward pass; optional per-clip forwards reduce peak activation memory."""
    if not micro_batch or imgs.shape[0] <= 1:
        return model(imgs)
    return torch.cat([model(imgs[i : i + 1]) for i in range(imgs.shape[0])], dim=0)


def train(
    config: TrainConfig,
    train_list: list[str] | None = None,
    test_list: list[str] | None = None,
) -> Path:
    """
    Train an RppgModel on H5 clips (Contrast-Phys+ protocol).

    When train_list/test_list are omitted, splits are derived from config
    (val_subjects). Split metadata is saved to split.json.
    """
    if train_list is None or test_list is None:
        train_list, test_list = resolve_h5_splits(config)

    device = resolve_device(config.device)
    use_amp = config.use_amp and supports_amp(device)
    print(f"Training on {device_label(device)} (AMP={'on' if use_amp else 'off'})", flush=True)
    run_dir = Path(
        os.path.join(
            config.checkpoint_dir,
            f"facecrop={config.face_crop_mode}_clipsec={config.clip_seconds}_epochs={config.epochs}_lr={config.lr}_imgsz={config.face_size}_backbone={config.model}",
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    ensure_mlflow()
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": config.model,
                "in_ch": config.in_ch,
                "fs": config.fs,
                "clip_seconds": config.clip_seconds,
                "spatial_dim": config.spatial_dim,
                "epochs": config.epochs,
                "lr": config.lr,
                "label_ratio": config.label_ratio,
                "face_size": config.face_size,
                "face_crop_mode": config.face_crop_mode,
                "h5_dir": resolve_h5_dir(config),
                "use_amp": use_amp,
                "grad_checkpoint": config.grad_checkpoint,
                "micro_batch": config.micro_batch,
                "device": device_label(device),
            }
        )

        split_path = save_split_metadata(run_dir, config)
        mlflow.log_artifact(str(split_path))
        mlflow.log_artifact(str(config_path))

        dataset = H5ClipDataset(
            train_list, config.T, config.label_ratio, face_size=config.face_size
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        model = build_model(
            config.model,
            spatial_dim=config.spatial_dim,
            in_ch=config.in_ch,
            grad_checkpoint=config.grad_checkpoint,
        )
        model = model.to(device).train()

        delta_t = config.T // 2
        loss_fn = ContrastLoss(delta_t, k=4, fs=config.fs, high_pass=40, low_pass=250)
        ipr_fn = IrrelevantPowerRatio(fs=config.fs, high_pass=40, low_pass=250)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        iters_per_epoch = int(np.round(config.video_duration_sec / (config.T / config.fs)))

        global_step = 0
        for epoch in tqdm(range(config.epochs), desc="Training epochs", unit="epoch"):
            for _ in range(iters_per_epoch):
                for imgs, gt_sig, label_flag in loader:
                    imgs = imgs.to(device, non_blocking=device.type == "cuda")
                    gt_sig = gt_sig.to(device, non_blocking=device.type == "cuda")
                    label_flag = label_flag.to(device, non_blocking=device.type == "cuda")

                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        output = _forward_model(model, imgs, config.micro_batch)
                        loss, p_loss, n_loss, p_gt, n_gt = loss_fn(output, gt_sig, label_flag)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    rppg = output[:, -1].float()
                    ipr = torch.mean(ipr_fn(rppg.detach()))

                    mlflow.log_metrics(
                        {
                            "loss": loss.item(),
                            "p_loss": p_loss.item(),
                            "n_loss": n_loss.item(),
                            "p_loss_gt": p_gt.item(),
                            "n_loss_gt": n_gt.item(),
                            "ipr": ipr.item(),
                        },
                        step=global_step,
                    )
                    global_step += 1

            ckpt = run_dir / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
            mlflow.log_artifact(str(ckpt))

    return run_dir
