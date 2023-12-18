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

"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from evaluation.metrics import utils
import numpy as np
from six.moves import range
import gin
from .utils import pca_torch

@gin.configurable(
    "factor_vae_score",
    denylist=["ground_truth_data", "representation_function", "random_state"])
def compute_factor_vae(ground_truth_data,
                       representation_function,
                       random_state,
                       batch_size=gin.REQUIRED,
                       num_train=gin.REQUIRED,
                       num_eval=gin.REQUIRED,
                       num_variance_estimate=gin.REQUIRED,
                       group_label=gin.REQUIRED,
                       pca_dim=0):
  """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  logging.info("Computing global variances to standardise.")
  observations = ground_truth_data.sample_observations(num_variance_estimate, random_state)
  representations = utils.obtain_representation(observations,
                                                representation_function,
                                                batch_size=64)
  representations = representations.astype('float64')
  group_label_unique = np.unique(group_label)
  mus_train1 = np.zeros((len(group_label_unique) * pca_dim, representations.shape[1]))
  eigenvectors_list = []
  for i, g in enumerate(group_label_unique):
      mus_train_g = representations[group_label == g, :]
      mus_train1[i * pca_dim:(i + 1) * pca_dim, :], eigenvectors = \
          pca_torch(mus_train_g, pca_dim, return_eigenvectors=True)
      eigenvectors_list.append(eigenvectors)
  representations = mus_train1

  global_variances = _compute_variances(representations.T,
                                        num_variance_estimate, random_state)
  active_dims = _prune_dims(global_variances)
  scores_dict = {}

  if not active_dims.any():
    scores_dict["train_accuracy"] = 0.
    scores_dict["eval_accuracy"] = 0.
    scores_dict["num_active_dims"] = 0
    return scores_dict

  logging.info("Generating training set.")
  training_votes = _generate_training_batch(ground_truth_data,
                                            representation_function, batch_size,
                                            num_train, random_state,
                                            global_variances, active_dims, eigenvectors=eigenvectors_list) # shape of training_votes is factor_num x active_dims
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])
  # list_classifier = list(classifier)
  # list_classifier = [int(e) for e in list_classifier]
  # scores_dict["classifier"] = list_classifier

  logging.info("Evaluate training set accuracy.")
  train_accuracy = np.sum(
      training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
  logging.info("Training set accuracy: %.2g", train_accuracy)

  logging.info("Generating evaluation set.")
  if ground_truth_data.__class__.__name__ == 'Market1501Attribute':
      random_state = 'test'
      num_eval = len(ground_truth_data.images['test'])
  eval_votes = _generate_training_batch(ground_truth_data,
                                        representation_function, batch_size,
                                        num_eval, random_state,
                                        global_variances, active_dims, eigenvectors=eigenvectors_list)

  logging.info("Evaluate evaluation set accuracy.")
  eval_accuracy = np.sum(eval_votes[classifier,
                                    other_index]) * 1. / np.sum(eval_votes)
  logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  scores_dict["num_active_dims"] = len(active_dims)
  return scores_dict


@gin.configurable("prune_dims", denylist=["variances"])
def _prune_dims(variances, threshold=0.):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold


def _compute_variances(representations,
                       batch_size,
                       random_state,
                       eval_batch_size=64,
                       ):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.
    eval_batch_size: Batch size used to eval representation.

  Returns:
    Vector with the variance of each dimension.
  """
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(ground_truth_data, representation_function,
                              batch_size, random_state, global_variances,
                              active_dims, eigenvectors=None):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
  """
  # Select random coordinate to keep fixed.
  factor_index = random_state.randint(ground_truth_data.num_factors)
  # Sample two mini batches of latent variables.
  factors = ground_truth_data.sample_factors(batch_size, random_state)
  # Fix the selected factor across mini-batch.
  factors[:, factor_index] = factors[0, factor_index]
  # Obtain the observations.
  observations = ground_truth_data.sample_observations_from_factors(
      factors, random_state)
  representations = representation_function(observations)
  representations = representations.astype('float64').T
  if eigenvectors is not None:
      if isinstance(eigenvectors, list):
          representations_split = np.array_split(representations, len(eigenvectors), axis=0)
          representations_list = []
          for e, r in zip(eigenvectors, representations_split):
              representations_list.append(e[1] @ (r - e[0]))
          representations = np.concatenate(representations_list, axis=0)
      else:
        representations = eigenvectors[1] @ (representations - eigenvectors[0])
  local_variances = np.var(representations.T, axis=0, ddof=1)
  argmin = np.argmin(local_variances[active_dims] /
                     global_variances[active_dims])
  return factor_index, argmin


def _generate_training_batch(ground_truth_data, representation_function,
                             batch_size, num_points, random_state,
                             global_variances, active_dims, eigenvectors=None):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
  votes = np.zeros((ground_truth_data.num_factors, global_variances.shape[0]),
                   dtype=np.int64)
  for _ in range(num_points):
    factor_index, argmin = _generate_training_sample(ground_truth_data,
                                                     representation_function,
                                                     batch_size, random_state,
                                                     global_variances,
                                                     active_dims, eigenvectors=eigenvectors)
    votes[factor_index, argmin] += 1
  return votes
