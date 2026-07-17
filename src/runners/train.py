import torch
from tqdm import tqdm
import wandb

from utils.gpu_setup import all_reduce_sum, is_main, train_dev_break
from utils.runner_helpers import batch_to_device

def run_train(
    nn,
    optimizer,
    dataloader,
    epoch,
    args,
    checkpoint_manager=None,
    ema=None,
):
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    show_progress = is_main()

    total_loss = 0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Training: {args.neural_network}; Task: {args.task};Epoch: {epoch}",
        disable=not show_progress,
        leave=False,
    )
    total_steps_per_epoch = len(dataloader)
    device = next(nn.parameters()).device

    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}

        optimizer.zero_grad()
        out = nn(**batch)
        loss = out.loss
        total_loss += loss.item()
        total_steps += 1
        loss.backward()
        grad_clip = getattr(args, "grad_clip", 0.0)
        if grad_clip > 0:
            params = (p for p in nn.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step_and_update_lr()
        if ema is not None:
            ema.update()
        if getattr(args, "wandb", False) and is_main():
            wandb.log({"train/step_loss": loss.item(), "train/lr": optimizer.learning_rate, "epoch": epoch})
        if args.save_step and checkpoint_manager and is_main():
            if checkpoint_manager.save_step(step, total_steps_per_epoch):
                checkpoint_manager.save_checkpoint(nn, optimizer, epoch, step, prefix="step_", ema=ema)
        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break
        # if step > 4000:
        #     break

    total_loss = all_reduce_sum(total_loss)
    total_steps = all_reduce_sum(total_steps)
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    if getattr(args, "wandb", False) and is_main():
        wandb.log({"train/loss": average_loss, "epoch": epoch})
    return {"average_loss": average_loss, "total_steps": total_steps}


@torch.no_grad()
def run_validation(nn, dataloader, epoch, args):
    if dataloader is None:
        return None
    nn.eval()
    device = next(nn.parameters()).device
    total_loss = 0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Validating: {args.neural_network}; Task: {args.task};Epoch: {epoch}",
        disable=not is_main(),
        leave=False,
    )
    for batch in progress:
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}
        loss = nn(**batch).loss
        total_loss += loss.item()
        total_steps += 1
        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break

    nn.train()
    total_loss = all_reduce_sum(total_loss)
    total_steps = all_reduce_sum(total_steps)
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    if getattr(args, "wandb", False) and is_main():
        wandb.log({"val/loss": average_loss, "epoch": epoch})
    return average_loss
