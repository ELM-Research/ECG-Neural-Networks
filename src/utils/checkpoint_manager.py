import torch
import os
from utils.gpu_setup import is_main


class CheckpointManager:
    def __init__(self, run_dir, args):
        self.run_dir = run_dir
        self.args = args
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.best_loss = float("inf")
        self.epoch_losses = []
        if is_main():
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step, is_best=False, prefix="", ema=None):

        filename = f"{prefix}epoch_{epoch}_step_{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if self.args.distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
        torch.save(checkpoint, filepath)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{prefix}best.pt")
            torch.save(checkpoint, best_path)

    def save_epoch(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.epoch_losses.append(loss)
            return True
        self.epoch_losses.append(loss)
        return False

    def save_step(self, step, total_steps_per_epoch):
        if step == 0:
            return True
        save_interval = max(1, total_steps_per_epoch // 5)
        return step % save_interval == 0

    def stop_early(self):
        if len(self.epoch_losses) < self.args.patience + 1:
            return False
        best_loss = min(self.epoch_losses[: -self.args.patience])
        current_loss = min(self.epoch_losses[-self.args.patience :])
        return current_loss > best_loss - self.args.patience_delta
    
    def load_checkpoint(self, model, optimizer, path, ema=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        (model.module if self.args.distributed else model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_loss = ckpt.get("best_loss", float("inf"))
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if is_main():
            print(f"Resumed from {path} at epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1