#!/usr/bin/env python3
"""
train_swin.py

Train a Swin model on ImageNet-style data (train/val directories in ImageFolder layout).

Usage example:
    python train_swin.py \
      --data-dir /data/imagenet \
      --variant tiny \
      --input-size 224 \
      --batch-size 256 \
      --epochs 90 \
      --workers 8 \
      --output-dir ./checkpoints

Notes:
 - Assumes you have either:
     from swin_factory import swin_tiny, swin_small, swin_base
   or your own SwinTransformer + factory in the same PYTHONPATH.
 - For multi-GPU training, run under torch.distributed.launch or adapt this script.
"""

import argparse
import math
import os
import time
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from swin_backbone import swin_tiny, swin_small, swin_base

# ----------------------------
# Utilities: Mixup, accuracy, warmup scheduler
# ----------------------------
def accuracy_topk(output, target, topk=(1,)):
    """Computes the top-k accuracy for specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res  # list values matching topk order

class Mixup:
    """Simple Mixup implementation"""
    def __init__(self, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5, label_smoothing=0.0, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def __call__(self, x, y):
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:
            return x, y, None, None  # no mix
        lam = 1.0
        use_cutmix = False
        if torch.rand(1).item() < self.prob:
            if self.cutmix_alpha > 0 and torch.rand(1).item() < self.switch_prob:
                # cutmix
                lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
                use_cutmix = True
            else:
                lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
                use_cutmix = False
            batch_size = x.size(0)
            perm = torch.randperm(batch_size).to(x.device)
            y_perm = y[perm]
            if use_cutmix:
                # Compute bounding box
                _, _, H, W = x.shape
                cut_rat = math.sqrt(1.0 - lam)
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                # uniform center
                cx = torch.randint(W, (1,)).item()
                cy = torch.randint(H, (1,)).item()
                x1 = max(cx - cut_w // 2, 0)
                x2 = min(cx + cut_w // 2, W)
                y1 = max(cy - cut_h // 2, 0)
                y2 = min(cy + cut_h // 2, H)
                x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            else:
                x = lam * x + (1 - lam) * x[perm]
            # Create one-hot labels with smoothing if needed
            if self.label_smoothing > 0:
                off_value = self.label_smoothing / (self.num_classes - 1)
                on_value = 1.0 - self.label_smoothing
                y_onehot = torch.full((y.size(0), self.num_classes), off_value, device=y.device)
                y_onehot.scatter_(1, y.unsqueeze(1), on_value)
                y_perm_onehot = torch.full((y_perm.size(0), self.num_classes), off_value, device=y.device)
                y_perm_onehot.scatter_(1, y_perm.unsqueeze(1), on_value)
                y_mix = lam * y_onehot + (1 - lam) * y_perm_onehot
                return x, y, y_mix, lam
            else:
                return x, y, (y, y_perm, lam), lam
        else:
            return x, y, None, None

class CosineWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, total_epochs, steps_per_epoch, warmup_epochs=5, min_lr=1e-6, last_epoch=-1):
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # linear warmup from 0 to base_lr
            return [base_lr * (float(step + 1) / float(max(1, self.warmup_steps))) for base_lr in self.base_lrs]
        else:
            # cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

# ----------------------------
# Multi-scale transform
# ----------------------------
class RandomMultiScaleResize:
    def __init__(self, scales=(224, 256, 288, 320), interpolation=InterpolationMode.BICUBIC):
        self.scales = list(scales)
        self.interp = interpolation

    def __call__(self, img):
        target = int(float(self.scales[torch.randint(low=0, high=len(self.scales), size=(1,)).item()]))
        w, h = img.size
        # preserve aspect ratio: scale by factor so the longer side equals target
        if h >= w:
            new_h = target
            new_w = int(round(w * (target / float(h))))
        else:
            new_w = target
            new_h = int(round(h * (target / float(w))))
        return transforms.functional.resize(img, (new_h, new_w), self.interp)

# ----------------------------
# Build dataloaders
# ----------------------------

def prepare_ilsvrc_paths(ilsvrc_root: Path):
    """
    Given the ILSVRC root Path (the folder named 'ILSVRC' in your find output),
    return (train_dir, val_dir, ann_dir) for:
      train -> ILSVRC/Data/CLS-LOC/train
      val   -> ILSVRC/Data/CLS-LOC/val
      ann   -> ILSVRC/Annotations/CLS-LOC (or ILSVRC/Annotations/CLS-LOC/val)
    Raises ValueError if expected structure not found.
    """
    il = Path(ilsvrc_root)
    data_cls_loc = il / "Data" / "CLS-LOC"
    ann_cls_loc = il / "Annotations" / "CLS-LOC"

    train_dir = data_cls_loc / "train"
    val_dir = data_cls_loc / "val"

    if not train_dir.exists():
        raise ValueError(f"Train dir not found: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Val dir not found: {val_dir}")
    if not ann_cls_loc.exists():
        # sometimes annotations for val are under Annotations/CLS-LOC/val
        if (il / "Annotations" / "CLS-LOC" / "val").exists():
            ann_cls_loc = il / "Annotations" / "CLS-LOC" / "val"
        else:
            raise ValueError(f"Annotations dir not found: {ann_cls_loc}")

    return train_dir, val_dir, ann_cls_loc

def build_dataloaders(data_dir, input_size=224, batch_size=256, num_workers=8,
                      multiscale_scales=(224, 256, 288), augment=True):
    """
    Expects `data_dir` to be the path to the ILSVRC folder (the folder you showed in your find output).
    It will use:
      data_dir/Data/CLS-LOC/train
      data_dir/Data/CLS-LOC/val
      data_dir/Annotations/CLS-LOC/*.xml
    If val is flat, it creates data_dir/val_by_class with symlinks and uses that for validation.
    Returns: train_loader, val_loader, num_classes
    """
    data_dir = Path(data_dir)

    # locate expected ILSVRC paths
    train_dir, val_dir, ann_dir = prepare_ilsvrc_paths(data_dir)

    # transforms (reuse RandomMultiScaleResize defined earlier in your script)
    if augment:
        train_transform = transforms.Compose([
            RandomMultiScaleResize(scales=multiscale_scales),
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset.classes)

# ----------------------------
# Training / validation loops
# ----------------------------
def validate(model, val_loader, device):
    model.eval()
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validate", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            top1, top5 = accuracy_topk(outputs, targets, topk=(1,5))
            bs = targets.size(0)
            total_top1 += top1 * bs / 100.0
            total_top5 += top5 * bs / 100.0
            total_samples += bs
    if total_samples == 0:
        return 0.0, 0.0
    top1 = total_top1 / total_samples * 100.0
    top5 = total_top5 / total_samples * 100.0
    return top1, top5

def train_one_epoch(model, optimizer, scaler, train_loader, device, epoch,
                    mixup_fn=None, criterion=None, grad_clip=None, print_freq=50, scheduler=None):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0
    it = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixup handling
        if mixup_fn is not None:
            mixed_x, orig_y, mixup_labels, lam = mixup_fn(images, targets)
            if mixup_labels is not None:
                pass
            images = mixed_x

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            if mixup_fn is not None and mixup_fn.label_smoothing > 0 and isinstance(mixup_labels, torch.Tensor):
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -(mixup_labels * log_probs).sum(dim=1).mean()
            elif mixup_fn is not None and mixup_labels is not None and isinstance(mixup_labels, tuple):
                y1, y2, lam = mixup_labels
                loss = lam * F.cross_entropy(outputs, y1) + (1 - lam) * F.cross_entropy(outputs, y2)
            else:
                loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            # step by iteration
            scheduler.step()

        # metrics
        bs = targets.size(0)
        total_samples += bs
        running_loss += loss.item() * bs
        top1, top5 = accuracy_topk(outputs.detach(), targets, topk=(1,5))
        running_top1 += top1 * bs / 100.0
        running_top5 += top5 * bs / 100.0

        it += 1
        if it % print_freq == 0:
            avg_loss = running_loss / total_samples
            avg_top1 = running_top1 / total_samples * 100.0
            avg_top5 = running_top5 / total_samples * 100.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", top5=f"{avg_top5:.2f}")

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_top1 = running_top1 / total_samples * 100.0 if total_samples > 0 else 0.0
    avg_top5 = running_top5 / total_samples * 100.0 if total_samples > 0 else 0.0
    return avg_loss, avg_top1, avg_top5

# ----------------------------
# Main training entry
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin on ImageNet-style dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="ImageNet root folder (must contain train/ and val/)")
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base"], help="Swin variant")
    parser.add_argument("--input-size", type=int, default=224, help="Input crop size")
    parser.add_argument("--multiscale", nargs="+", type=int, default=[224, 256, 288], help="Multi-scale sizes for RandomMultiScaleResize")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=None, help="If absent, inferred from dataset")
    parser.add_argument("--pretrained", action="store_true", help="If you have pretrained weights to load (not implemented auto-download)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--sync-bn", action="store_true", help="Convert BatchNorm to SyncBatchNorm (if using DistributedDataParallel)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, num_classes_inferred = build_dataloaders(
        args.data_dir, input_size=args.input_size, batch_size=args.batch_size,
        num_workers=args.workers, multiscale_scales=args.multiscale, augment=True
    )
    num_classes = args.num_classes if args.num_classes is not None else num_classes_inferred

    # Create model
    if args.variant == "tiny":
        model = swin_tiny(num_classes=num_classes)
    elif args.variant == "small":
        model = swin_small(num_classes=num_classes)
    else:
        model = swin_base(num_classes=num_classes)

    model.to(device)

    # Optionally resume
    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            start_epoch = ckpt.get("epoch", 0)
            best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed checkpoint from {args.resume} epoch {start_epoch}")

    # Optimizer, criterion, scheduler
    # weight decay on all parameters except LayerNorm / bias often
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.shape) == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # Steps per epoch for scheduler
    steps_per_epoch = len(train_loader)
    total_epochs = args.epochs
    scheduler = CosineWithWarmupLR(optimizer, total_epochs=total_epochs, steps_per_epoch=steps_per_epoch, warmup_epochs=args.warmup_epochs)

    # Criterion with label smoothing
    if args.label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Mixup
    mixup_fn = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, prob=1.0, label_smoothing=args.label_smoothing, num_classes=num_classes)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    for epoch in range(start_epoch, total_epochs):
        # training
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, optimizer, scaler, train_loader, device, epoch,
            mixup_fn=mixup_fn, criterion=criterion, grad_clip=args.grad_clip, print_freq=args.print_freq,
            scheduler=scheduler
        )
        # validation
        val_top1, val_top5 = validate(model, val_loader, device)

        is_best = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_top1": best_top1,
            "args": vars(args),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch+1}.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best_checkpoint.pt")

        print(f"Epoch {epoch+1}/{total_epochs}  train_loss={train_loss:.4f}  train_top1={train_top1:.2f}  val_top1={val_top1:.2f}  best_top1={best_top1:.2f}")

    # Save final model
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "final_model.pt")
    print("Training finished. Checkpoints and final_model.pt saved in", out_dir)

if __name__ == "__main__":
    main()
