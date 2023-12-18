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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from evaluation.metrics import utils
import scipy
from six.moves import range
import gin
import numpy as np
from .utils import pca_torch
from sklearn.ensemble import GradientBoostingClassifier

@gin.configurable(
    "dci",
    denylist=["ground_truth_data", "representation_function", "random_state"])
def compute_dci(ground_truth_data, representation_function, random_state,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                lr=gin.REQUIRED,
                batch_size=16,
                group_label=None,
                feature_cached=False,
                pca_dim=0,
                factor_num=-1,
                ):
  """Computes the DCI scores according to Sec 2.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with average disentanglement score, completeness and
      informativeness (train and test).
  """
  logging.info("Generating training set.")
  # the shape of mus_train is [num_codes, num_train], the shape of ys_train is [num_factors, num_train].
  if feature_cached:
    rep_dict = representation_function
    mus_train = rep_dict['mus_train']
    mus_test = rep_dict['mus_test']
    if ground_truth_data is not None:
      ys_train = ground_truth_data.factors['train'].T
      ys_test = ground_truth_data.factors['test'].T
    else:
      ys_train = rep_dict['ys_train']
      ys_test = rep_dict['ys_test']
  else:
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train,
        random_state, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test,
        random_state, batch_size)

  mus_train = mus_train.astype(np.float64)
  mus_test = mus_test.astype(np.float64)
  eigen_vector_list = []
  mean_list = []
  group_label_unique = np.unique(group_label)
  mus_train1 = np.zeros((len(group_label_unique) * pca_dim, mus_train.shape[1]))
  mus_test1 = np.zeros((len(group_label_unique) * pca_dim, mus_test.shape[1]))
  for i, g in enumerate(group_label_unique):
      mus_train_g = mus_train[group_label == g, :]
      mus_test_g = mus_test[group_label == g, :]
      mus_train1[i*pca_dim:(i+1)*pca_dim, :], mus_test1[i*pca_dim:(i+1)*pca_dim, :], pca_transform = \
      pca_torch(mus_train_g, pca_dim, mus_test_g, return_eigenvectors=True)
      mean_list.append(pca_transform[0])
      eigen_vector_list.append(pca_transform[1][:, :, np.newaxis])
  mus_train, mus_test = mus_train1, mus_test1
  scores, importance_matrix = _compute_dci(mus_train, ys_train, mus_test, ys_test, lr, factor_num)
  return scores, importance_matrix


def _compute_dci(mus_train, ys_train, mus_test, ys_test, dci_lr, factor_num):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test, dci_lr, factor_num)
  if importance_matrix is not None:
      assert importance_matrix.shape[0] == mus_train.shape[0]
      assert importance_matrix.shape[1] == ys_train.shape[0]
      scores["disentanglement"] = disentanglement(importance_matrix)
      scores["completeness"] = completeness(importance_matrix)
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  return scores, importance_matrix


def compute_importance_gbt(x_train, y_train, x_test, y_test, dci_lr, factor_num):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []

  for i in range(num_factors):
    if factor_num != -1 and i != factor_num:
      continue
    model = GradientBoostingClassifier(learning_rate=dci_lr)
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
