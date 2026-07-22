import torch
from tqdm import tqdm
import numpy as np

from utils.gpu_setup import is_main
from utils.viz import plot_forecast, plot_forecast_trajectories
from utils.eval_stats import forecast_metrics
from utils.runner_helpers import batch_to_device
from utils.dir_file import DirFileManager
from configs.constants import PTB_ORDER

def eval_forecasting(nn, dataloader, args):
    return eval_forecasting_bpe(nn, dataloader, args)

def eval_forecasting_bpe(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Forecasting",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    data_repr = dataloader.dataset.data_representation
    condition_lead = "_".join(map(str, args.condition_lead))
    condition_name = f"{args.condition}_{condition_lead}" if args.condition else f"{args.condition}"
    data_names = "_".join(args.data)
    plot_dir = f"{args.run_dir}/{data_names}_{args.forecast_ratio}_{args.bpe_symbolic_len}_{condition_name}_{args.lead_tokens}"
    DirFileManager.ensure_directory_exists(folder=plot_dir)
    all_acc, all_sig = [], []
    max_seq_len = args.bpe_symbolic_len

    if args.condition:
        n_leads = len(args.condition_lead)
        lead_names = [PTB_ORDER[i] for i in args.condition_lead]
    else:
        n_leads = len(PTB_ORDER)
        lead_names = PTB_ORDER
    n_total = n_leads * args.segment_len

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            labels = batch["labels"].numpy()
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            tgt_ids = batch["tgt_ids"]
            if tgt_ids.size(1) >= max_seq_len:
                tgt_ids = tgt_ids[:, -(max_seq_len - 1):]
            pred = nn.generate(tgt_ids, max_new_tokens=labels.shape[1],
                               return_new_only=True, return_logits=False).cpu().numpy()
            gt = labels[:, :pred.shape[1]]
            all_acc.append((pred == gt).mean())
            for i in range(len(gt)):
                mn, mx = batch["min"][i].item(), batch["max"][i].item()
                ps = data_repr.decode(pred[i].tolist(), mn, mx)
                gs = data_repr.decode(gt[i].tolist(), mn, mx)
                ctx = data_repr.decode(tgt_ids[i].tolist(), mn, mx)
                all_sig.append(forecast_metrics(ps, gs, context=np.array(ctx)))
                if step < 10 and i == 0:
                    ctx = data_repr.decode(tgt_ids[0].tolist(), mn, mx)
                    n_ctx = len(ctx)
                    n_gt_end = min(len(ctx) + len(gs), n_total)
                    n_pred_end = min(len(ctx) + len(ps), n_total)
                    full_gt = np.pad(np.concatenate([ctx, gs]), (0, max(0, n_total - len(ctx) - len(gs))))[:n_total].reshape(n_leads, args.segment_len)
                    full_pred = np.pad(np.concatenate([ctx, ps]), (0, max(0, n_total - len(ctx) - len(ps))))[:n_total].reshape(n_leads, args.segment_len)
                    plot_forecast(full_gt, full_pred, n_ctx, n_gt_end, n_pred_end,
                                  report, f"{plot_dir}/plot_{step}.png",
                                  segment_len=args.segment_len, leads=lead_names, sf=args.sf)
                    if args.n_forecast_trajectories > 1:
                        trajs = [full_pred]
                        for _ in range(args.n_forecast_trajectories - 1):
                            sampled = nn.generate(tgt_ids, max_new_tokens=labels.shape[1],
                                                  return_new_only=True, return_logits=False, do_sample=True).cpu().numpy()
                            s = data_repr.decode(sampled[0].tolist(), mn, mx)
                            full_s = np.pad(np.concatenate([ctx, s]), (0, max(0, n_total - len(ctx) - len(s))))[:n_total]
                            trajs.append(full_s.reshape(n_leads, args.segment_len))
                        plot_forecast_trajectories(full_gt, np.stack(trajs), n_ctx,
                                                   report, f"{plot_dir}/traj_{step}.png",
                                                   segment_len=args.segment_len, leads=lead_names, sf=args.sf)
            if step > 200:
                break

    metrics = {"accuracy": float(np.nanmean(all_acc))}
    for k in all_sig[0]:
        metrics[k] = float(np.nanmean([s[k] for s in all_sig]))
    print("Forecast | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics