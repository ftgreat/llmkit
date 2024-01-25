# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import numpy as np
import torch

class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights, size, *,
                 data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indicies.
        def _build_indices():
            start_time = time.time()
            assert num_datasets < 255
            dataset_index = np.zeros(self.size, dtype=np.uint8)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from megatron.data import helpers
            helpers.build_blending_indices(dataset_index, dataset_sample_index,
                                           weights, num_datasets, self.size,
                                           torch.distributed.get_rank() == 0)
            print('> elapsed time for building blendable dataset indices: '
                         '{:.2f} (sec)'.format(time.time() - start_time), flush=True)
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc

        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")

            # Load on all ranks.
            print(f'> loading blendable dataset index: {index_path}', flush=True)
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_index.size == self.size

            print(f'> loading blendable dataset sample index: {sample_index_path}', flush=True)
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_sample_index.size == self.size
        else:
            self.dataset_index, self.dataset_sample_index = _build_indices()


        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError('BlendedDataset size is improperly bounded')
        except IndexError:
            pass
        print('> size of blendable dataset: '
                     '{} samples'.format(self.size), flush=True)


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx" : dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }
