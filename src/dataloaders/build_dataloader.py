import argparse

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.gpu_setup import get_world_size, get_rank

from dataloaders.dataset_mixer import DatasetMixer


class BuildDataLoader:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.args = args
        self.dataset_mixer = DatasetMixer(self.args)
        self.val_dataloader = None

    def build_dataloader(
        self,
    ):
        train_dataset, val_dataset = self.dataset_mixer.build_torch_dataset()
        torch_data_loader = self.build_torch_dataloader(train_dataset)
        if val_dataset is not None:
            self.val_dataloader = self.build_torch_dataloader(val_dataset, is_val=True)
        return torch_data_loader

    def build_torch_dataloader(self, torch_dataset, is_val=False):
        sampler = self.get_torch_dataloader_sampler(torch_dataset, shuffle=not is_val)
        if "train" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=(sampler is None and not is_val),
                num_workers=self.args.num_workers,
                sampler=sampler,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
                persistent_workers=(self.args.num_workers > 0),
                prefetch_factor=4 if self.args.num_workers > 0 else None,
            )
        elif "eval" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=1,  # batched inference/eval not implemented
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
            )
        return torch_data_loader

    def get_torch_dataloader_sampler(
        self,
        torch_dataset,
        shuffle=True,
    ):
        if self.args.distributed:
            sampler = DistributedSampler(torch_dataset, num_replicas=get_world_size(), 
                                         rank=get_rank(), seed=self.args.seed, shuffle=shuffle)
        else:
            sampler = None
        return sampler

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
