"""Falcon training pipeline with AdamW, AMP, and CosineAnnealing scheduler."""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from falcon.config import ModelConfig
from falcon.data.dataset import build as build_data
from falcon.losses import AgeGenderLoss, OrdinalAgeLoss, WeightedMSE
from falcon.model.factory import create_model
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_logger = logging.getLogger("train")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Falcon Training")
    parser.add_argument("--dataset-images", type=str, required=True)
    parser.add_argument("--dataset-annotations", type=str, required=True)
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["utk", "imdb", "lagenda", "fairface", "adience", "agedb", "cacd"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", type=str, default="", help="pretrained VOLO checkpoint")
    parser.add_argument("--output", type=str, default="output/train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--with-persons", action="store_true")
    parser.add_argument("--num-age-bins", type=int, default=101)
    parser.add_argument(
        "--age-mode",
        choices=["regression", "distribution", "ordinal"],
        default="distribution",
    )
    return parser


def create_loss_fn(age_mode: str, num_age_bins: int, only_age: bool = False) -> nn.Module:
    """Build the loss function based on the age head mode."""
    if age_mode == "distribution":
        return AgeGenderLoss(num_age_bins=num_age_bins, only_age=only_age)
    if age_mode == "ordinal":
        return OrdinalAgeLoss(num_classes=num_age_bins)
    return WeightedMSE()


def train_epoch(model, loader, criterion, optimizer, scaler, device, half, epoch):
    """Run a single training epoch.

    Args:
        model: The model to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: AMP gradient scaler (or ``None``).
        device: Torch device.
        half: Whether to use FP16.
        epoch: Current epoch number (for logging).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    start = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if half:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            _logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / n_batches
    _logger.info(f"Epoch {epoch} avg_loss: {avg_loss:.4f}, time: {time.time() - start:.1f}s")
    return avg_loss


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.age_mode == "distribution":
        num_classes = 2 + args.num_age_bins
    elif args.age_mode == "ordinal":
        num_classes = args.num_age_bins
    else:
        num_classes = 1

    config = ModelConfig(
        with_persons_model=args.with_persons,
        use_persons=args.with_persons,
        num_classes=num_classes,
        device=torch.device(args.device),
    )

    dataset, loader = build_data(
        name=args.dataset_name,
        images_path=args.dataset_images,
        annotations_path=args.dataset_annotations,
        split=args.split,
        model_config=config,
        workers=4,
        batch_size=args.batch_size,
    )

    model = create_model(
        model_name="falcon_d1_224",
        num_classes=num_classes,
        in_chans=config.in_chans,
        pretrained=False,
        checkpoint_path=args.checkpoint,
        filter_keys=["fds."],
    )
    model = model.to(config.device)

    criterion = create_loss_fn(args.age_mode, args.num_age_bins)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.half else None

    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            config.device,
            args.half,
            epoch,
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint-{epoch}.pth.tar")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": loss,
                    "min_age": dataset.min_age,
                    "max_age": dataset.max_age,
                    "avg_age": dataset.avg_age,
                    "no_gender": False,
                    "with_persons_model": args.with_persons,
                },
                ckpt_path,
            )
            _logger.info(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
