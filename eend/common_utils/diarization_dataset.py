#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import common_utils.features as features
import common_utils.kaldi_data as kaldi_data
import numpy as np
import torch
from typing import Tuple
import h5py
import os


def _count_frames(data_len: int, size: int, step: int) -> int:
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
    data_length: int,
    size: int,
    step: int,
    use_last_samples: bool,
    min_length: int,
) -> None:
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step > min_length:
            yield (i + 1) * step, data_length


def get_h5_info(feat_h5_idx):
    file_set = set()
    utt2len_filename = {}

    with open(feat_h5_idx, 'r') as f:
        for line in f.readlines():
            utt, num_frames, h5_path = line.strip().split()
            num_frames = int(num_frames)
            file_set.add(h5_path)

            utt2len_filename[utt] = (num_frames, '/'.join(h5_path.split('/')[-2:]))
    return utt2len_filename, file_set


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        context_size: int,
        feature_dim: int,
        frame_shift: int,
        frame_size: int,
        input_transform: str,
        n_speakers: int,
        sampling_rate: int,
        shuffle: bool,
        subsampling: int,
        use_last_samples: bool,
        min_length: int,
        dtype: type = np.float32,
        read_feat: bool = False,
    ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.feature_dim = feature_dim
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.sampling_rate = sampling_rate
        self.chunk_indices = []
        self.read_feat = read_feat  # directly read feat from h5 file

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(
                self.data.reco2dur[rec] * sampling_rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            if chunk_size > 0:
                for st, ed in _gen_frame_indices(
                        data_len,
                        chunk_size,
                        chunk_size,
                        use_last_samples,
                        min_length
                ):
                    self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
            else:
                self.chunk_indices.append(
                    (rec, 0, data_len * self.subsampling))

        self.shuffle = shuffle

        self.h5_f_dict = None
        if self.read_feat:
            self.utt2len_filename, self.h5_file_set = get_h5_info(os.path.join(self.data_dir, 'feat_h5_idx'))
        else:
            self.utt2len_filename, self.h5_file_set = None, None

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.read_feat and self.h5_f_dict is None:
            self.h5_f_dict = {}
            for h5_file_path in self.h5_file_set:
                h5_file_name = '/'.join(h5_file_path.split('/')[-2:])
                self.h5_f_dict[h5_file_name] = h5py.File(h5_file_path, 'r')

        rec, st, ed = self.chunk_indices[i]

        if self.read_feat:
            num_frames, h5_file_name = self.utt2len_filename[rec]
            assert ed <= num_frames, 'Recording {} requires end at {} but only has {} frames.'.format(rec, ed, num_frames)
            Y = self.h5_f_dict[h5_file_name][rec][st:ed]
            Y, T = features.get_labels(
                Y,
                self.data,
                rec,
                st,
                ed,
                self.frame_size,
                self.frame_shift,
                self.n_speakers,
                rate=self.sampling_rate
            )
            Y = features.norm_log_mel(Y, self.input_transform)
        else:
            Y, T = features.get_labeledSTFT(
                self.data,
                rec,
                st,
                ed,
                self.frame_size,
                self.frame_shift,
                self.n_speakers
            )
            Y = features.transform(
                Y, self.sampling_rate, self.feature_dim, self.input_transform)
        Y_spliced = features.splice(Y, self.context_size)
        Y_ss, T_ss = features.subsample(Y_spliced, T, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        #  extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]

        return torch.from_numpy(np.copy(Y_ss)), torch.from_numpy(
            np.copy(T_ss)), rec
