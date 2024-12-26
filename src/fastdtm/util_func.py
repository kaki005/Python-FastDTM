from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.core import ShapedArray
from jax.scipy.linalg import cho_factor, cho_solve, cholesky
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar
from polyagamma import random_polyagamma

from .polyagamma import sample_pg_int

ZERO = 1e-6


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, jnp.eye(P.shape[-1]))


# JAX対応のclip関数
@jax.jit
def jax_exp(x):
    return jnp.clip(jnp.exp(jnp.clip(x, a_min=-700, a_max=700)), a_min=ZERO)


@eqx.filter_jit
def softmax_multi_posterior(
    mean: jnp.ndarray, cov: jnp.ndarray, count: jnp.ndarray, key: PRNGKeyArray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mean = mean.squeeze()
    N = jnp.sum(count, dtype=jnp.double)
    keys = random.split(key, count.shape[0])
    N_list = jnp.concat((jnp.expand_dims(N, axis=0), N - jnp.cumsum(count)[:-1]))
    omega = sample_polyagamma(N_list.astype(int), mean)
    cov_inv = inv(cov)
    new_cov = inv(jnp.diag(omega) + cov_inv)
    new_mean = new_cov @ (count - N_list / 2 + cov_inv @ mean)
    return new_mean, new_cov


@eqx.filter_vmap
@eqx.filter_jit
def sample_polyagamma(N: Int[Scalar, "1"], mean: Float[Scalar, "1"]) -> Float[Scalar, "1"]:
    result_shape = ShapedArray((), dtype=jnp.float32)
    # return jax.pure_callback(partial(random_polyagamma, method="devroye"), result_shape, N, mean)
    is_positive = N > 0
    N = is_positive * N + (1 - is_positive) * (N + 3e-4)
    return jax.pure_callback(random_polyagamma, result_shape, N, mean)


@eqx.filter_jit
@eqx.filter_vmap
def _perplexity_vmap(llh: Float[Scalar, "1"], word_num) -> Float[Scalar, "1"]:
    return jnp.exp(-llh / word_num)
