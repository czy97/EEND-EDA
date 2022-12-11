#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from scipy.optimize import linear_sum_assignment


def pit_loss_multispk(
        logits: List[torch.Tensor], target: List[torch.Tensor],
        n_speakers: np.ndarray):

    logits_t = logits.detach().transpose(1, 2)
    cost_mxs = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target)

    max_n_speakers = max(n_speakers)

    for i, cost_mx in enumerate(cost_mxs.cpu().numpy()):
        if max_n_speakers > n_speakers[i]:
            max_value = np.absolute(cost_mx).sum()
            cost_mx[-(max_n_speakers-n_speakers[i]):] = max_value
            cost_mx[:, -(max_n_speakers-n_speakers[i]):] = max_value
        # use Hungarian algorithm to solve Bipartite graph problem
        pred_alig, ref_alig = linear_sum_assignment(cost_mx)
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        target[i, :] = target[i, :, ref_alig]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
             logits, target, reduction='none')

    loss_mask = (target != -1) * 1.0
    loss = loss * loss_mask
    loss = torch.sum(loss) / torch.sum(loss_mask)

    return loss, target


def vad_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    vad_ts = (torch.sum((ts != -1)*ts, 2) > 0).float()  # (B, T)
    # We work on the probability space, not logits. We use silence probabilities
    ys_silence_probs = 1-torch.sigmoid(ys)
    all_ones_probs = torch.ones_like(ys_silence_probs)
    ys_silence_probs = torch.where(ts < 0, all_ones_probs, ys_silence_probs)
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2)  # (B, T)
    loss = F.binary_cross_entropy(silence_prob, 1-vad_ts, reduction='none')
    loss_mask = ((ts != -1).sum(dim=2) > 0).float()
    loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
    return loss

