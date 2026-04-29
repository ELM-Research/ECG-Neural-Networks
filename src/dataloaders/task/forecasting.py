import numpy as np
import torch
from typing import Tuple, Any, Dict


class Forecasting:
    def __init__(self, args):
        self.args = args

    def __call__(self, transformed_data):
        if self.args.data_representation == "bpe_symbolic":
            return self.bpe_symbolic(transformed_data)
        elif self.args.data_representation == "signal":
            return self.signal(transformed_data)

    def signal(self, transformed_data: Dict[str, Any]):
        inputs = np.asarray(transformed_data["transformed_data"])
        if "eval" in self.args.mode:
            context_len = int(inputs.shape[-1] * (1 - self.args.forecast_ratio))
            return {"signal": inputs, "context_len": context_len, "report": transformed_data.get("report", "")}
        padding_mask = transformed_data.get("padding_mask")
        modality_mask = transformed_data.get("modality_mask")
        if self.args.objective == "autoregressive":
            return {
                "signal": inputs,
                "padding_mask": padding_mask,
                "modality_mask": modality_mask,
            }

    def bpe_symbolic(self, transformed_data: Dict[str, Any]):
        inputs = np.asarray(transformed_data["transformed_data"])
        orig_len = inputs.shape[0]
        labels = inputs.copy()

        if "train" in self.args.mode and self.args.objective == "autoregressive":
            labels = self.autoregressive(labels)
        else:
            labels, inputs = self.prepare_eval(inputs, labels)

        inputs = torch.as_tensor(inputs, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None


        if self.args.neural_network == "trans_discrete_decoder":
            out = {"tgt_ids": inputs, "labels": labels}
            if "eval" in self.args.mode:
                out.update({"min": transformed_data["min"], "max": transformed_data["max"],
                            "report": transformed_data["report"], "orig_len" : orig_len})
            if getattr(self.args, "signal_head", False):
                out["signal"] = torch.as_tensor(transformed_data["signal"], dtype=torch.float32)
            return out

    def autoregressive(self, labels: np.ndarray) -> np.ndarray:
        non_pad_mask = labels != self.args.pad_id
        if non_pad_mask.any():
            first_non_pad = non_pad_mask.argmax()
            non_pad_len = non_pad_mask.sum()
            num_to_mask = int(non_pad_len * (1 - self.args.forecast_ratio))
            labels[first_non_pad : first_non_pad + num_to_mask] = -100
        labels[labels == self.args.pad_id] = -100
        labels[labels == self.args.bos_id] = -100
        return labels

    def prepare_eval(self, inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        split_idx = int(len(labels) * (1 - self.args.forecast_ratio))
        return labels[split_idx:], inputs[:split_idx]