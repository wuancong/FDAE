# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
from evaluation.metrics import utils
import numpy as np
import gin

import cv2
import matplotlib.pyplot as plt


def plot_histograms(data_in):
    data = data_in.T
    n, d = data.shape
    fig, axs = plt.subplots(d)
    for i in range(d):
        axs[i].hist(data[:, i], bins=50)
    plt.show()


def z_score_normalize(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    arr = (arr - arr_mean) / arr_std
    arr = (arr + 1) / 2 * 255
    arr = np.clip(arr, 0, 255)
    return arr


def max_min_normalize(arr):
    arr_max, arr_min = np.max(arr), np.min(arr)
    arr = (arr - arr_min) / (arr_max - arr_min)
    arr *= 255
    return arr


def histeq(arr):
  """
  对每一个维度做直方图均衡化
  """
  arr = arr.astype(np.float64)
  for i in range(arr.shape[0]):
    arr[i, :] = max_min_normalize(arr[i, :])
  arr = arr.astype(np.uint8)
  out = np.zeros_like(arr)
  for i in range(arr.shape[0]):
    out[i, :] = cv2.equalizeHist(np.reshape(arr[i, :], (arr.shape[1], 1)))[:, 0]
  out = out.astype(np.float64)
  return out


@gin.configurable("mig", denylist=["ground_truth_data", "representation_function", "random_state"])
def compute_mig(ground_truth_data,
                representation_function,
                random_state,
                num_train=gin.REQUIRED,
                batch_size=16,
                group_label=None,
                cluster_method=None,
                ):
  """Computes the mutual information gap.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
  logging.info("Generating training set.")
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  mus_train = mus_train.astype(np.float64)
  if group_label is not None:
    output = _compute_group_mig(mus_train, ys_train, group_label, cluster_method)
  else:
    output = _compute_mig(mus_train, ys_train)
  return output


def _compute_mig(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  discretized_mus = utils.make_discretizer(mus_train)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  assert m.shape[0] == mus_train.shape[0]
  assert m.shape[1] == ys_train.shape[0]
  # m: shape num_latents x num_factors
  entropy = utils.discrete_entropy(ys_train)
  sorted_m = np.sort(m, axis=0)[::-1]
  score_dict["discrete_mig"] = np.mean(
      np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
  return score_dict


def _compute_group_mig(mus_train, ys_train, group_label, cluster_method):
  """
  group_label is np.array of the same length as the latents, indicating the group of concept
  Computes score based on both training and testing codes and factors.
  """
  score_dict = {}
  discretized_mus = utils.make_discretizer(mus_train, group_label=group_label, cluster_method=cluster_method)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  entropy = utils.discrete_entropy(ys_train)
  sorted_m = np.sort(m, axis=0)[::-1]
  score_dict["discrete_mig"] = np.mean(
      np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
  return score_dict