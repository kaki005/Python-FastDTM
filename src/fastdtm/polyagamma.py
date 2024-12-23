from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from jax.random import exponential, gamma, split, uniform
from jaxtyping import Bool, Float, Int, PRNGKeyArray, Scalar

TRUNC = 2 / jnp.pi
TRUNC_INV = 1.0 / TRUNC


# Section 3
# J∗(1,z)を生成するj_star_one_samplerを作成。
# J*(n,z)はJ*(1,z)の独立なサンプルを合計。
def sample_pg_int(key: PRNGKeyArray, h: Int[Scalar, "1"], z: Float[Scalar, "1"]):
    """
    Direct sampler for J*(n, z) using Devroye method for J*(1, z).
    Parameters:
        key: jax.random.PRNGKey
        n: int, shape parameter
        z: float, tilting parameter
    Returns:
        Sample from J*(n, z).
    """
    # key, subkey = random.split(key)
    # keys = random.split(subkey, h.squeeze())
    init = (key, 0.0, 0)

    def _loop_body(carry):
        key, ret, index = carry
        a, key = sample_J1z(key, h)  # sample
        ret += a
        return (key, ret, index + 1)

    def _loop_cond(carry):
        _, _, index = carry
        return index <= h.squeeze()

    _, ret, _ = jax.lax.while_loop(_loop_cond, _loop_body, init)
    return ret


def sample_J1z(key: PRNGKeyArray, z: Float[Scalar, "1"]):
    z = jnp.abs(z) * 0.5
    K = 0.125 * jnp.pi * jnp.pi + 0.5 * z * z  # π^2/8 + z^2/2
    # ... Problems with large Z?  Try using q_over_p.
    # p = 0.5 * jnp.pi * jnp.exp(-1.0 * K * TRUNC) / K
    # q = 2 * jnp.exp(-1.0 * z) * pigauss(TRUNC, z)
    carry_main = (key, 0.0, False)

    def sample_loop(carry_main):
        key, _, _ = carry_main
        # sample from X ~ g(x|z)
        key, unif = _uniform(key)

        def sample_L(key):
            # sample from truncated Exponential
            subkey, key = random.split(key)
            X = TRUNC + jax.random.exponential(subkey) / K
            return X, key

        def sample_R(key):
            key, X = rtigauss(z, key)
            return X, key

        X, key = jax.lax.cond(unif < mass_texpon(z), sample_L, sample_R, key)
        key, unif = _uniform(key)
        S = a(0, X)
        U = unif * S  # sample from uniform(0, a_0(X))
        carry = (0, True, S, False)

        def calc_S_body(carry):
            n, go, s, finish = carry

            def calc_S_body_odd(carry):
                n, go, s, finish = carry
                s -= a(n, X)
                finish = U <= s
                return (n + 1, go, s, finish)

            def calc_S_body_even(carry):
                n, go, s, finish = carry
                s += a(n, X)
                go = U <= s
                return (n + 1, go, s, finish)

            return jax.lax.cond(n % 2 == 1, calc_S_body_odd, calc_S_body_even, carry)

        def calc_S_cond(carry):
            n, go, s, finish = carry
            return finish

        n, go, _, _ = jax.lax.while_loop(calc_S_cond, calc_S_body, carry)
        return (key, X, go)

    def sample_cond(carry_main):
        key, _, go = carry_main
        return go

    key, X, _ = jax.lax.while_loop(sample_cond, sample_loop, carry_main)  # return J^*(1, z)
    return X, key


def a(n: Int[Scalar, "1"] | int, x: Float[Scalar, "1"]):
    """compute a_n(x)."""
    K = (n + 0.5) * jnp.pi  # π*(n+ 1/2)
    y = jnp.select(
        [x > TRUNC, x > 0],
        [
            K * jnp.exp(-0.5 * K * K * x),
            jnp.exp(-1.5 * (jnp.log(0.5 * jnp.pi) + jnp.log(x)) + jnp.log(K) - 2.0 * (n + 0.5) * (n + 0.5) / x),
        ],
    )
    return y


def mass_texpon(z: Float[Scalar, "1"]):
    """calc p /(q+p)."""
    t = TRUNC
    fz = 0.125 * jnp.pi * jnp.pi + 0.5 * z * z  # π^2/8 + z^2/2
    b = jnp.sqrt(1.0 / t) * (t * z - 1)
    a = jnp.sqrt(1.0 / t) * (t * z + 1) * -1.0
    x0 = jnp.log(fz) + fz * t
    xb = x0 - z + jnp.log(jax.scipy.stats.norm.cdf(b))  # cdf of normal gaussian proces
    xa = x0 + z + jnp.log(jax.scipy.stats.norm.cdf(a))
    qdivp = 4 / jnp.pi * (jnp.exp(xb) + jnp.exp(xa))
    return 1.0 / (1.0 + qdivp)


def pigauss(x: Float[Scalar, "1"] | float, z: Float[Scalar, "1"]):
    """CDF of the inverse Gaussian distribution."""
    b = jnp.sqrt(1.0 / x) * (x * z - 1)
    a = jnp.sqrt(1.0 / x) * (x * z + 1) * -1.0
    y = jax.scipy.stats.norm.cdf(b) + jnp.exp(2 * z) * jax.scipy.stats.norm.cdf(a)
    return y


def rtigauss(Z, key: PRNGKeyArray):
    """sample from truncated inverse gaussian."""
    Z = jnp.abs(Z)
    t = TRUNC
    X = t + 1.0

    def sample_chi2(key: PRNGKeyArray):
        r"""Generate X ~ χ^2_1・1_{(t,\infty)}."""
        key, unif = _uniform(key)
        carry1 = (key, X)

        def _sample_chi2_loop(carry1):
            key, _ = carry1
            carry = (key, 0, 0)

            def _loop_body1(carry):
                key, E1, E2 = carry
                key, E1 = _exponential(key)
                key, E2 = _exponential(key)
                return (key, E1, E2)

            def _loop_cond1(carry) -> Bool:
                key, E1, E2 = carry
                return E1 * E1 > 2 * E2 / t

            key, E1, E2 = jax.lax.while_loop(_loop_cond1, _loop_body1, carry)
            X = 1 + E1 * t
            X = t / (X * X)
            return key, X

        def _sample_chi2cond(carry1) -> Bool:
            key, X = carry1
            alpha = jnp.exp(-0.5 * Z * Z * X)  # exp(-z^2/2 X)
            return unif > alpha

        return jax.lax.while_loop(_sample_chi2cond, _sample_chi2_loop, carry1)

    def sample_inverse_gaussian(key: PRNGKeyArray):
        r"""Generate X ~ IN(\mu, 1.0)."""
        mu = 1.0 / Z
        carry = (key, X)

        def _loop_body2(carry):
            key, _ = carry
            key, Y = _normal(key, 1.0, 1.0)
            Y *= Y
            half_mu = 0.5 * mu
            mu_Y = mu * Y
            X = mu + half_mu * mu_Y - half_mu * jnp.sqrt(4 * mu_Y + mu_Y * mu_Y)
            key, unif = _uniform(key)
            X = jax.lax.select(unif > mu / (mu + X), mu * mu / X, X)
            return (key, X)

        def _loop_cond2(carry):
            key, X = carry
            return X > t

        return jax.lax.while_loop(_loop_cond2, _loop_body2, carry)

    return jax.lax.cond(TRUNC_INV > Z, sample_chi2, sample_inverse_gaussian, key)


def _normal(key: PRNGKeyArray, mean, std) -> tuple[PRNGKeyArray, Float[Scalar, "1"]]:
    key, subkey = jax.random.split(key)
    return key, mean + random.normal(subkey) * std


def _uniform(key: PRNGKeyArray) -> tuple[PRNGKeyArray, Float[Scalar, "1"]]:
    key, subkey = jax.random.split(key)
    return key, uniform(subkey)


def _exponential(key: PRNGKeyArray) -> tuple[PRNGKeyArray, Float[Scalar, "1"]]:
    key, subkey = jax.random.split(key)
    return key, exponential(subkey)
