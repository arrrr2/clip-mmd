# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Thanks to 
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

"""Memory-efficient MMD implementation in JAX."""

import torch
import numpy as np


# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )


    return _SCALE * (k_xx + k_yy - 2 * k_xy)


def kernel_matrix_ce(x, x_sqnorms, y, y_sqnorms, gamma, bs=128):
    n_x = x.shape[0]
    n_y = y.shape[0]
    x_sqnorms = x_sqnorms
    y_sqnorms = y_sqnorms

    kernel_sum = 0.0

    for l in range(0, n_x, bs):

        r = min(l + bs, n_x)
        x_batch = x[l:r, :]
        x_sqnorms_batch = x_sqnorms[l:r]

        
        dists = x_sqnorms_batch[:, None] + y_sqnorms[None, :] - 2 * torch.matmul(x_batch, y.t())
        kernel_values = torch.exp(-gamma * dists)
        kernel_sum += kernel_values.sum()

    kernel_mean = kernel_sum / (n_x * n_y)
    return kernel_mean.cpu()


def mmd_efficient(x, y, device='cpu', precision=torch.float32, low_mem=False, coeff_bs=128):

    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    x = x.to(device).to(precision)
    y = y.to(device).to(precision)

    import time
    t = time.perf_counter()
    
    x_sqnorms = torch.sum(x**2, axis=1)
    y_sqnorms = torch.sum(y**2, axis=1)
    
    
    gamma = 1 / (2 * _SIGMA**2)
    
    if low_mem:
        k_xx = kernel_matrix_ce(x, x_sqnorms, x, x_sqnorms, gamma, coeff_bs)
        k_yy = kernel_matrix_ce(y, y_sqnorms, y, y_sqnorms, gamma, coeff_bs)
        k_xy = kernel_matrix_ce(x, x_sqnorms, y, y_sqnorms, gamma, coeff_bs)
    else:
        k_xx = torch.exp(-gamma * (x_sqnorms[:, None] + x_sqnorms[None, :] - 2 * torch.mm(x, x.t()))).mean()
        k_yy = torch.exp(-gamma * (y_sqnorms[:, None] + y_sqnorms[None, :] - 2 * torch.mm(y, y.t()))).mean()
        k_xy = torch.exp(-gamma * (x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * torch.mm(x, y.t()))).mean()
    
    
    mmd_value = (_SCALE * (k_xx + k_yy - 2 * k_xy)).cpu()

    return mmd_value


