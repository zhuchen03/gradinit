# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from tqdm import tqdm


class Stat:
    def __init__(self, keep_raw=False):
        self.x = 0.0
        self.x2 = 0.0
        self.z = 0.0  # z = logx
        self.z2 = 0.0
        self.n = 0.0
        self.u = 0.0
        self.keep_raw = keep_raw
        self.raw = []

    def update(self, new_x):
        new_z = new_x.log()

        self.x += new_x.sum()
        self.x2 += (new_x**2).sum()
        self.z += new_z.sum()
        self.z2 += (new_z**2).sum()
        self.n += len(new_x)
        self.u += 1

        if self.keep_raw:
            self.raw.append(new_x)

    @property
    def mean(self):
        return self.x / self.n

    @property
    def std(self):
        return (self.x2 / self.n - self.mean**2) ** 0.5

    @property
    def mean_log(self):
        return self.z / self.n

    @property
    def std_log(self):
        return (self.z2 / self.n - self.mean_log**2) ** 0.5

    @property
    def n_frms(self):
        return self.n

    @property
    def n_utts(self):
        return self.u

    @property
    def raw_data(self):
        assert self.keep_raw, "does not support storing raw data!"
        return torch.cat(self.raw)


class F0Stat(Stat):
    def update(self, new_x):
        # assume unvoiced frames are 0 and consider only voiced frames
        if new_x is not None:
            super().update(new_x[new_x != 0])


def dump_speaker_f0_stat(speaker_to_f0_stat, out_prefix):
    path = f"{out_prefix}.f0_stat.pt"
    assert not os.path.exists(path)

    d = {
        speaker: {
            "f0_mean": speaker_to_f0_stat[speaker].mean,
            "f0_std": speaker_to_f0_stat[speaker].std,
            "logf0_mean": speaker_to_f0_stat[speaker].mean_log,
            "logf0_std": speaker_to_f0_stat[speaker].std_log,
        }
        for speaker in speaker_to_f0_stat
    }
    torch.save(d, path)

    return d


def load_audio_path(path):
    audio_paths = []
    with open(path) as f:
        for line in f.readlines():
            sample = eval(line.strip())
            audio_paths.append(sample["audio"])

    return audio_paths


def load_f0(f0_dir, nshards):
    path_to_f0 = {}
    for rank in tqdm(range(1, nshards + 1), desc=f"load f0"):
        f0_shard_path = f"{f0_dir}/f0_{rank}_{nshards}.pt"
        shard_path_to_f0 = torch.load(f0_shard_path)
        path_to_f0.update(shard_path_to_f0)
    return path_to_f0
