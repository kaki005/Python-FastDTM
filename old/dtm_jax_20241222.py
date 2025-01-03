import itertools
import logging
import sqlite3
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import softmax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar

from src.configs.config import ModelConfig

ZERO = 1e-6


# JAX対応のclip関数
@jax.jit
def jax_exp(x):
    return jnp.clip(jnp.exp(jnp.clip(x, a_min=-700, a_max=700)), a_min=ZERO)


class DTMState(NamedTuple):
    Z: list[list[Float[Array, "W"]]]
    """topic assignment. (time, document at t, word)"""
    CDK: list[Float[Array, "D K"]]
    """counter. (time, num of documnt at t, topic)"""
    CWK: Float[Array, "T V K"]
    """counter. (T, V, K)"""
    CK: Float[Array, "T K"]
    """counter.(time, topic)"""
    alpha: Float[Array, "T K"]
    """time evolving parameters topic latent strengh. (time, topic)"""
    phi: Float[Array, "T W K"]
    """time evolving word strength for each topic. (time, word, topic)"""
    eta: list[Float[Array, "D K"]]
    """topic strengh with uncertainity. (time, num of document at t, topic)"""
    key: PRNGKeyArray
    # term_alias_samples = np.zeros((self.T, self.V, self.K), dtype=int)
    # """buffer of random value created from alias table. (time, word, topic)"""


class DTMJax(eqx.Module):
    W: list[list[Float[Array, "word_num"]]] = eqx.field(static=True)
    """data (time, document at t, wotd index)"""
    vocabulary: list[str] = eqx.field(static=True)
    """list of word in data."""
    logger: logging.Logger = eqx.field(static=True)
    K: int = eqx.field(static=True)
    """num of topic"""
    V: int = eqx.field(static=True)
    """num of words"""
    T: int = eqx.field(static=True)
    """num of time"""
    D: jnp.ndarray = eqx.field(static=True)
    """num of document at each time (time, document)"""
    word_nums: list[Float[Array, "doc_num_t"]]
    """num of words for each document. (time, document)"""
    sgld_a: float = eqx.field(static=True)
    sgld_b: float = eqx.field(static=True)
    sgld_c: float = eqx.field(static=True)
    dtm_phi_var: float = eqx.field(static=True)
    """variance of phi."""
    dtm_eta_var: float = eqx.field(static=True)
    """variance of eta."""
    dtm_alpha_var: float = eqx.field(static=True)
    """variance of alpha."""
    index: eqx.nn.StateIndex

    def __init__(self, W, vocabulary, config: ModelConfig):
        self.W = [[jnp.array(wtd) for wtd in w_t] for w_t in W]
        self.word_nums = [jnp.array([len(w_td) for w_td in w_t]) for w_t in W]
        self.vocabulary = vocabulary
        self.logger = logging.getLogger(str(__class__))
        self.K = config.num_topic
        self.V = len(self.vocabulary)
        self.T = len(self.W)
        self.D = jnp.array([len(w_t) for w_t in W])
        self.sgld_a = config.SGLD_a
        self.sgld_b = config.SGLD_b
        self.sgld_c = config.SGLD_c
        self.dtm_phi_var = config.phi_var
        self.dtm_eta_var = config.eta_var
        self.dtm_alpha_var = config.alpha_var
        # 初期化
        Z = [[jnp.zeros(len(self.W[t][d]), dtype=int) for d in range(self.D[t])] for t in range(self.T)]
        CDK = [jnp.zeros((self.D[t], self.K), dtype=int) for t in range(self.T)]
        CWK = jnp.zeros((self.T, self.V, self.K), dtype=int)
        CK = jnp.zeros((self.T, self.K), dtype=int)
        alpha = jnp.zeros((self.T, self.K), dtype=float)
        phi = jnp.zeros((self.T, self.V, self.K), dtype=float)
        eta = [jnp.zeros((self.D[t], self.K)) for t in range(self.T)]
        key = random.key(0)
        self.index = eqx.nn.StateIndex(DTMState(Z, CDK, CWK, CK, alpha, phi, eta, key))

    def initialize(self, model_state: eqx.nn.State, init_with_lda=True) -> eqx.nn.State:
        init_alpha = 50.0 / self.K
        init_beta = 0.01
        Z, CDK, CWK, CK, alpha, phi, eta, key = model_state.get(self.index)

        for t in range(self.T):
            key, subkey = random.split(key)
            keys = random.split(subkey, self.D[t])
            for d in range(self.D[t]):
                N = len(self.W[t][d])
                keys_d = random.split(keys[d], N)
                for n in range(N):
                    w = self.W[t][d][n]
                    k = random.randint(keys_d[n], (1,), 0, self.K - 1).squeeze()  # 初期トピック
                    Z[t][d] = Z[t][d].at[n].set(k)
                    CDK[t] = CDK[t].at[d, k].add(1)
                    CWK = CWK.at[t, w, k].add(1)
                    CK = CK.at[t, k].add(1)
                    eta[t] = eta[t].at[d, k].add((1 + init_alpha) / (N + self.K * init_alpha))
        if init_with_lda:  # t=0のみLDA
            for iter in range(50):
                for d in range(self.D[0]):
                    N = len(self.W[0][d])
                    for n in range(N):
                        k = Z[0][d][n]
                        w = self.W[0][d][n]
                        CDK[t] = CDK[t].at[d, k].subtract(1)
                        CWK = CWK.at[t, w, k].subtract(1)
                        CK = CK.at[t, k].subtract(1)
                        prob = jnp.array(
                            [
                                CDK[0][d, k] + init_alpha * (CWK[0][w][k] + init_beta) / (CK[0][k] + self.V * init_beta)
                                for k in range(self.K)
                            ]
                        )
                        key, subkey = random.split(key)
                        k = random.categorical(subkey, jnp.log(prob))
                        Z[0][d] = Z[0][d].at[n].set(k)
                        CDK[0] = CDK[0].at[d, k].add(1)
                        CWK = CWK.at[0, w, k].add(1)
                        CK = CK.at[0, k].add(1)
        new_state = model_state.set(self.index, DTMState(Z, CDK, CWK, CK, alpha, phi, eta, key))
        return new_state

    def estimate(self, model_state: eqx.nn.State, num_iters: int) -> eqx.nn.State:
        Z, CDK, CWK, CK, alpha, phi, eta, key = model_state.get(self.index)

        # @eqx.filter_jit
        # @eqx.filter_vmap(in_axes=[None, 0, 0, 0, 0, 0, 0, 0, 0])
        def _sample_doc_loop(carry_doc, params):
            t, key, CDK_t, CWK, CK, eta_t, phi_t, Z_t = carry_doc
            d, N = params
            key, subkey = random.split(key)
            xi_vec = jnp.ones(self.K) * random.normal(subkey, (1,)) * eps
            # sample eta
            prior_eta = (alpha[t] - eta_t) / self.dtm_eta_var
            grad_eta = CDK_t - N * softmax(eta_t)  # (10)
            eta_t += eps / 2 * (prior_eta + grad_eta) + xi_vec  # (10)
            # sample topi
            carry = (key, CDK_t, CWK, CK, 0)

            def _sample_topic_word(carry, param):
                key, CDK_td, CWK, CK, word_index = carry
                pre_topic = param
                w = self.W[t][d][word_index]

                def _mh_test_word(key: PRNGKeyArray, pre_topic: int):
                    key, subkey = random.split(key)
                    index = random.randint(subkey, (1,), 0, N).squeeze()  # sample random word
                    proposal = Z_t[index]
                    acceptance_prob = jax_exp(phi_t[w, proposal]) / jax_exp(phi_t[w, pre_topic])
                    return proposal, acceptance_prob, key

                def _mh_test_topic(key: PRNGKeyArray, pre_topic: int):
                    key, subkey = random.split(key)
                    proposal = random.randint(subkey, (1,), 0, self.K - 1).squeeze()  # sample random word
                    acceptance_prob = jax_exp(eta[proposal]) / jax_exp(eta[pre_topic])
                    return proposal, acceptance_prob, key

                def _sample(key: PRNGKeyArray, is_mh_word: bool, pre_topic: int, CDK_td, CWK, CK):
                    # metropolis hasting test.
                    CDK_td = CDK_td[pre_topic].subtract(1)
                    CWK = CWK.at[t, w, pre_topic].subtract(1)
                    CK = CK.at[t, pre_topic].subtract(1)
                    proposal, acceptance_prob, key = jax.lax.cond(is_mh_word, _mh_test_word, _mh_test_topic, key)
                    key, subkey = random.split(key)
                    is_rejected = random.uniform(subkey) >= acceptance_prob
                    new_topic = is_rejected * pre_topic + (1 - is_rejected) * proposal
                    CDK_td = CDK_td[new_topic].add(1)
                    CWK = CWK.at[t, w, new_topic].add(1)
                    CK = CK.at[t, new_topic].add(1)
                    return CDK_td, new_topic, CWK, CK

                keys = random.split(key, 3)
                CDK_td, new_topic, CWK, CK = _sample(keys[0], True, CDK_td, pre_topic, CWK, CK)
                CDK_td, new_topic, CWK, CK = _sample(keys[1], False, CDK_td, new_topic, CWK, CK)
                return (keys[2], CDK_td, CWK, CK, word_index + 1), new_topic

            (key, CDK_t, CWK, CK, _), Z_t[d] = jax.lax.scan(_sample_topic_word, carry, Z_t[d])
            return (t, key, CDK_t, CWK, CK, eta_t, phi_t, Z_t), None

        for i in range(num_iters):
            eps = self.sgld_a * jnp.pow(self.sgld_b + i, -self.sgld_c)
            for t in range(self.T):
                # region (sample eta, topic)
                carry = (t, key, CDK[t], CWK[t], CK[t], eta[t], phi[t], Z[t])
                params = (jnp.arange(self.D[t]), self.word_nums[t])
                (_, key, CDK[t], CWK[t], CK[t], eta[t], phi[t], Z[t]), _ = jax.lax.scan(_sample_doc_loop, carry, params)
                # endregion (sample eta, topic)

            # region (sample phi)

            def _sample_phi_loop(carry, param):
                key, phi, eps = carry
                t, cwk_t, ck_t = param
                subkey, key = random.split(key)
                xi_vec = jnp.ones(self.V) * random.normal(subkey) * eps

                @eqx.filter_vmap(in_axes=[2, 1, 0])
                def _sample_phi_vmap(phi_k: Float[Array, "T N"], cw_tk: Float[Array, "N"], c_tk):
                    def _left():
                        phi_sigma = 1.0 / ((1.0 / 100) + (1 / self.dtm_phi_var))
                        prior_phi = phi_k[t + 1] * (phi_sigma / self.dtm_phi_var)
                        return ((2 * prior_phi) - 2 * phi_k[t]) / self.dtm_phi_var

                    def _right():
                        return (phi_k[t - 1] - phi_k[t]) / self.dtm_phi_var

                    def _middle():
                        return (phi_k[t + 1] + phi_k[t - 1] - 2 * phi_k[t]) / self.dtm_phi_var

                    prior_phi = jax.lax.switch([t == 0, t == self.T - 1], [_left(), _right()], operand=_middle())
                    grad_phi = cw_tk - c_tk * softmax(phi_k[t])  # (14)
                    phi_k[t] += (eps / 2) * (grad_phi + prior_phi) + xi_vec  # (14)
                    return phi_k

                phi = _sample_phi_vmap(phi, cwk_t, ck_t)
                return (key, jnp.transpose(phi, (1, 2, 0)), eps), None

            carry = (key, phi, eps)
            (key, phi, _), _ = jax.lax.scan(_sample_phi_loop, carry, (jnp.arange(self.T), CWK, CK))

            # endregion (sample phi)
            # region (sample alpha)
            carry = (key, alpha)

            def _sample_alph_loop(carry, t: int):
                key, alpha = carry
                alpha_bar = jnp.zeros(self.K)

                def _left():
                    precision = (1.0 / 100) + (1 / self.dtm_alpha_var)
                    alpha_sigma = 1.0 / precision
                    alpha_bar = alpha[t + 1] * (alpha_sigma / self.dtm_alpha_var)
                    return precision, alpha_bar

                def _middle():
                    alpha_bar = (alpha[t - 1] - alpha[t]) / self.dtm_alpha_var
                    precision = 1.0 / self.dtm_alpha_var
                    return precision, alpha_bar

                def _right():
                    precision = 2 / self.dtm_alpha_var
                    alpha_bar = (alpha[t + 1] + alpha[t - 1]) / 2  # (5)
                    return precision, alpha_bar

                alpha_precision, alpha_bar = jax.lax.switch(
                    [t == 0, t == self.T - 1], [_left(), _right()], operand=_middle()
                )
                eta_bar = eta[t].mean(axis=0)  # (5)
                sigma_inv = 1.0 / (alpha_precision + self.D[t] / self.dtm_eta_var)  # (5)
                cov = jnp.eye(self.K) * sigma_inv  # (5)
                mean = (
                    alpha_bar + eta_bar - cov @ (eta_bar * alpha_precision + alpha_bar * self.D[t] / self.dtm_eta_var)
                )  # (4)
                key, subkey = random.split(key)
                alpha.at[t] = random.multivariate_normal(subkey, mean, cov)  # sample
                return (key, alpha)

            key, alpha = jax.lax.scan(_sample_alph_loop, carry, jnp.arange(self.T))
            # endregion (sample alpha)

        new_state = model_state.set(self.index, DTMState(Z, CDK, CWK, CK, alpha, phi, eta, key))
        return new_state
