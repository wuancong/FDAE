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

"""Utility functions that are useful for the different metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import sklearn.cluster as sk_cluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score as cluster_score
import gin
import torch


@gin.configurable(
    "pca_torch", denylist=["input_array", "test_array"])
def pca_torch(input_array: np.ndarray, n_components: int, test_array=None, return_eigenvectors=False,
              reg_w=0.0, device='cuda', print_flag=False):
  # 将输入的numpy数组转换为PyTorch张量
  input_array = input_array.T
  tensor = torch.from_numpy(input_array).to(device=device)
  input_mean = tensor.mean(dim=0)
  # 将输入张量中心化
  tensor = tensor - input_mean
  # 计算协方差矩阵
  cov_matrix = torch.mm(tensor.T, tensor) / (tensor.shape[0] - 1)
  cov_matrix = (cov_matrix + cov_matrix.T) / 2
  # 对协方差矩阵进行特征值分解
  eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
  eigenvalues = torch.real(eigenvalues)
  eigenvectors = torch.real(eigenvectors)
  # Sort the eigenvalues in descending order and get the sorted indices
  sorted_indices = torch.argsort(eigenvalues, descending=True)
  # Sort the eigenvectors using the sorted indices
  eigenvectors = eigenvectors[:, sorted_indices]
  if isinstance(n_components, int):
    # 获取前n个主成分
    eigenvectors = eigenvectors[:, :n_components]
    # 计算主成分能量占比
    eigenvalues = torch.real(eigenvalues)
    energy_ratio = eigenvalues[:n_components].sum() / eigenvalues.sum()
    if print_flag:
      print(energy_ratio)
  else:
    assert 0 <= n_components <= 1
    energy_ratio_cum = 0
    eigenvalues_sum = eigenvalues.sum()
    for i, e in enumerate(eigenvalues):
      energy_ratio_cum += e / eigenvalues_sum
      if energy_ratio_cum > n_components:
        n_components = i + 1
        break
    if print_flag:
      print(n_components)
    eigenvectors = eigenvectors[:, :n_components]
  # 将数据投影到主成分上
  result = torch.mm(tensor, eigenvectors)
  if test_array is not None:
    test_array = torch.from_numpy(test_array.T).to(device=device)
    test_array -= input_mean
    test_result = torch.mm(test_array, eigenvectors)
    if not return_eigenvectors:
      return result.cpu().numpy().T, test_result.cpu().numpy().T
    else:
      return result.cpu().numpy().T, test_result.cpu().numpy().T, (input_mean.cpu().numpy()[:, np.newaxis], eigenvectors.cpu().numpy().T)
  else:
    if not return_eigenvectors:
      return result.cpu().numpy().T
    else:
      return result.cpu().numpy().T, (input_mean.cpu().numpy()[:, np.newaxis], eigenvectors.cpu().numpy().T)


def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.

  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


@gin.configurable(
    "discretizer", denylist=["target"])
def make_discretizer(target, num_bins=gin.REQUIRED,
                     discretizer_fn=gin.REQUIRED,
                     **kwargs):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins, **kwargs)


@gin.configurable("histogram_discretizer", denylist=["target"])
def _histogram_discretize(target, num_bins=gin.REQUIRED, **kwargs):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


@gin.configurable("cluster_discretizer", denylist=["target"])
def _cluster_discretize(target, num_bins=gin.REQUIRED, random_state=gin.REQUIRED, return_model=False,
                        model_list_precomputed=None, discretize_type='cluster', **kwargs):
  """Discretization based on clustering."""
  pca_dim = kwargs.get('pca_dim')
  group_label = kwargs.get('group_label')

  group_label_unique = np.unique(group_label)
  if not (isinstance(num_bins, list) or isinstance(num_bins, tuple)):
    num_bins = [num_bins]
  if len(num_bins) == 1:
    num_bins *= len(group_label_unique)
  discretized = []

  for i, (g, num_bins_i) in enumerate(zip(group_label_unique, num_bins)):
    target_g = target[group_label == g, :]
    pca_result = pca_torch(target_g, pca_dim)
    if pca_dim == 1:
      for j in range(len(pca_result)):
        discretized_i = np.digitize(pca_result[j], np.histogram(pca_result[j], num_bins_i)[1][:-1])
        discretized.append(discretized_i[np.newaxis, :])
    else:
      kmeans = sk_cluster.KMeans(n_clusters=num_bins_i, random_state=random_state).fit(pca_result.T)
      discretized_i = kmeans.labels_
      discretized.append(discretized_i[np.newaxis, :])

  if isinstance(discretized, list):
    discretized = np.concatenate(discretized, axis=0)

  return discretized


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


@gin.configurable("predictor")
def make_predictor_fn(predictor_fn=gin.REQUIRED):
  """Wrapper that creates classifiers."""
  return predictor_fn


@gin.configurable("logistic_regression_cv")
def logistic_regression_cv():
  """Logistic regression with 5 folds cross validation."""
  return LogisticRegressionCV(Cs=10, cv=KFold(n_splits=5))


@gin.configurable("gradient_boosting_classifier")
def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return GradientBoostingClassifier()
