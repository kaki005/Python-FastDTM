import logging
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import softmax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar
from utilpy.jax import print_pytree
from wandb import init

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
    # ===========================
    # region (property)
    # ===========================
    W: list[list[Float[Array, "N_dt"]]] = eqx.field(static=True)
    """data (time, document at t, wotd index)"""
    Ns: list[Float[Array, "D_t"]]
    """num of words for each document. (time, document)"""
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
    # endregion (property)

    # ===========================
    # region (コンストラクタ,初期化)
    # ===========================

    def __init__(self, W, dictionary, config):
        self.W = [([jnp.array(w_td) for w_td in w_t]) for w_t in W]
        self.Ns = [jnp.array([len(w_td) for w_td in w_t]) for w_t in W]
        self.vocabulary = dictionary
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
        key = random.key(0)
        Z = []
        for t in range(self.T):
            Z.append([])
            for d in range(self.D[t]):
                key, subkey = random.split(key)
                Z[t].append(random.randint(subkey, (self.Ns[t][d],), 0, self.K - 1))
        CDK = [jnp.zeros((self.D[t], self.K), dtype=int) for t in range(self.T)]
        CWK = jnp.zeros((self.T, self.V, self.K), dtype=int)
        CK = jnp.zeros((self.T, self.K), dtype=int)
        key, subkey = random.split(key)
        alpha = random.normal(subkey, (self.T, self.K), dtype=float)
        key, subkey = random.split(key)
        phi = random.normal(subkey, (self.T, self.V, self.K), dtype=float)
        eta = [jnp.zeros((self.D[t], self.K)) for t in range(self.T)]
        self.index = eqx.nn.StateIndex(DTMState(Z, CDK, CWK, CK, alpha, phi, eta, key))

    def initialize(self, model_state: eqx.nn.State, init_with_lda=True) -> eqx.nn.State:
        init_alpha = 50.0 / self.K
        init_beta = 0.01
        Z, CDK, CWK, CK, alpha, phi, eta, key = model_state.get(self.index)

        for t in range(self.T):
            self.logger.info(f"{t}")
            key, subkey = random.split(key)
            keys = random.split(subkey, self.D[t])
            for d in range(self.D[t]):
                N = len(self.W[t][d])
                Z[t][d] = Z[t][d] = random.randint(keys[d], (N,), 0, self.K - 1)
                carry = (CDK[t], CWK, CK, eta[t], t, d, N)

                def _add_topic_loop(carry, params):
                    CDK_t, CWK, CK, eta_t, t, d, N = carry
                    w, k = params
                    CDK_t = CDK_t.at[d, k].add(1)
                    CWK = CWK.at[t, w, k].add(1)
                    CK = CK.at[t, k].add(1)
                    eta_t = eta_t.at[d, k].set((1 + init_alpha) / (N + self.K * init_alpha))
                    return (CDK_t, CWK, CK, eta_t, t, d, N), None

                (CDK[t], CWK, CK, eta[t], _, _, _), _ = jax.lax.scan(_add_topic_loop, carry, (self.W[t][d], Z[t][d]))

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

    # endregion(コンストラクタ,初期化)

    # ===========================
    # region (メソッド)
    # ===========================
    def estimate(self, model_state: eqx.nn.State, num_iters: int) -> eqx.nn.State:
        @eqx.filter_jit
        def _sample_eta(
            t: int,
            d: int,
            N: int,
            eps: float,
            eta_td: Float[Array, "K"],
            CDK_td: Float[Array, "K"],
            alpha_t: Float[Array, "K"],
        ):
            xi_vec = jnp.ones(self.K) * random.normal(key, (1,)) * eps
            prior_eta = (alpha_t - eta_td) / self.dtm_eta_var
            grad_eta = CDK_td - N * softmax(eta_td)  # (10)
            eta_td += (eps / 2) * (grad_eta + prior_eta) + xi_vec  # (10)
            return eta_td

        @eqx.filter_jit
        def _sample_topic(
            N: int,
            w: Int[Scalar, "1"],
            key: PRNGKeyArray,
            eta_td: Float[Array, "K"],
            # CDK_td: Float[Array, "K"],
            Z_td: Int[Scalar, "1"],
            phi: Float[Array, "V K"],
            Z_t: Int[Array, "N_td"],
        ):
            def _mh_test_word(key: PRNGKeyArray, pre_topic: Int[Scalar, "1"]):
                key, subkey = random.split(key)
                # index = random.randint(subkey, (1,), 0, N).squeeze()  # sample random word
                # proposal = Z_t[index]  # その単語のトピック
                proposal = random.randint(subkey, (1,), 0, self.K - 1).squeeze()  # sample random toipc
                acceptance_prob = jax_exp(phi[w, proposal]) / jax_exp(phi[w, pre_topic])
                return proposal, acceptance_prob, key

            def _mh_test_topic(key: PRNGKeyArray, pre_topic: Int[Scalar, "1"]):
                key, subkey = random.split(key)
                proposal = random.randint(subkey, (1,), 0, self.K - 1).squeeze()  # sample random toipc
                acceptance_prob = jax_exp(eta_td[proposal]) / jax_exp(eta_td[pre_topic])
                return proposal, acceptance_prob, key

            def _sample(key: PRNGKeyArray, is_mh_word: bool, pre_topic: Int[Scalar, "1"]):
                # metropolis hasting test.
                # CDK_td = CDK_td[pre_topic].subtract(1)
                proposal, acceptance_prob, key = jax.lax.cond(is_mh_word, _mh_test_word, _mh_test_topic, key, pre_topic)
                key, subkey = random.split(key)
                is_rejected = random.uniform(subkey) >= acceptance_prob
                new_topic = is_rejected * pre_topic + (1 - is_rejected) * proposal
                # CDK_td = CDK_td[new_topic].add(1)
                return new_topic

            keys = random.split(key, 3)
            new_topic = _sample(keys[0], True, Z_td)
            new_topic = _sample(keys[1], False, new_topic)
            return new_topic

        def _update_counter(t: int, d: int, CWK, CK, CDK_t, new_topic, old_topic):
            N = self.Ns[t][d]
            for n in range(N):
                w = self.W[t][d][n]
                old_k = old_topic[n]
                CWK = CWK.at[t, w, old_k].subtract(1)
                CK = CK.at[t, old_k].subtract(1)
                CDK_t = CDK_t.at[d, old_k].subtract(1)
                new_k = new_topic[n]
                CWK = CWK.at[t, w, new_k].add(1)
                CK = CK.at[t, new_k].add(1)
                CDK_t = CDK_t.at[d, new_k].add(1)
            return CWK, CK, CDK_t

        @eqx.filter_jit
        def _sample_alpha_loop(carry, params):
            t, eta_bar = params
            key, alpha = carry

            alpha_bar = jnp.zeros(self.K)

            def _left():
                precision = (1.0 / 100) + (1 / self.dtm_alpha_var)
                alpha_sigma = 1.0 / precision
                alpha_bar = alpha[t + 1] * (alpha_sigma / self.dtm_alpha_var)
                return alpha_bar

            def _middle():
                alpha_bar = (alpha[t - 1] - alpha[t]) / self.dtm_alpha_var
                # precision = 1.0 / self.dtm_alpha_var
                return alpha_bar

            def _right():
                # precision = 2 / self.dtm_alpha_var
                alpha_bar = (alpha[t + 1] + alpha[t - 1]) / 2  # (5)
                return alpha_bar

            alpha_precision = jnp.select(
                [t == 0, t == self.T - 1],
                [(1.0 / 100) + (1 / self.dtm_alpha_var), 1.0 / self.dtm_alpha_var],
                default=2 / self.dtm_alpha_var,
            )
            alpha_bar = jnp.select([t == 0, t == self.T - 1], [_left(), _right()], default=_middle())
            sigma_inv = 1.0 / (alpha_precision + self.D[t] / self.dtm_eta_var)  # (5)
            cov = jnp.eye(self.K) * sigma_inv  # (5)
            mean = (
                alpha_bar + eta_bar - cov @ (eta_bar * alpha_precision + alpha_bar * self.D[t] / self.dtm_eta_var)
            )  # (4)
            key, subkey = random.split(key)
            alpha = alpha.at[t].set(random.multivariate_normal(subkey, mean, cov))  # sample
            return (key, alpha), None

        for i in range(num_iters):
            Z, CDK, CWK, CK, alpha, phi, eta, key = model_state.get(self.index)
            eps = self.sgld_a * jnp.pow(self.sgld_b + i, -self.sgld_c).squeeze()
            for t in range(self.T):
                # region (sample eta)
                eta[t] = jax.vmap(_sample_eta, in_axes=[None, 0, 0, None, 0, 0, None])(
                    t, jnp.arange(self.D[t]), self.Ns[t], eps, eta[t], CDK[t], alpha[t]
                )
                # endregion (sample eta)

                # region (sample eta, topic)
                for d in range(self.D[t]):
                    subkey, key = random.split(key)
                    N = self.Ns[t][d]
                    keys = random.split(subkey, N)
                    new_topics = jax.vmap(_sample_topic, in_axes=[None, 0, 0, None, 0, None, None])(
                        N, self.W[t][d], keys, eta[t][d], Z[t][d], phi[t], Z[t]
                    )
                    CWK, CK, CDK[t] = _update_counter(t, d, CWK, CK, CDK[t], new_topics, Z[t][d])
                # endregion (sample eta, topic)

            # region (sample phi)
            def _sample_phi_loop(carry, param):
                key, phi, eps = carry
                t, cwk_t, ck_t = param
                subkey, key = random.split(key)
                xi_vec = jnp.ones(self.V) * random.normal(subkey) * eps

                def _sample_phi_vmap(phi_k: Float[Array, "T N"], cw_tk: Float[Array, "N"], c_tk):
                    def _left():
                        phi_sigma = 1.0 / ((1.0 / 100) + (1 / self.dtm_phi_var))
                        prior_phi = phi_k[t + 1] * (phi_sigma / self.dtm_phi_var)
                        return ((2 * prior_phi) - 2 * phi_k[t]) / self.dtm_phi_var

                    def _right():
                        return (phi_k[t - 1] - phi_k[t]) / self.dtm_phi_var

                    def _middle():
                        return (phi_k[t + 1] + phi_k[t - 1] - 2 * phi_k[t]) / self.dtm_phi_var

                    prior_phi = jnp.select([t == 0, t == self.T - 1], [_left(), _right()], default=_middle())
                    grad_phi = cw_tk - c_tk * softmax(phi_k[t])  # (14)
                    phi_k = phi_k.at[t].add((eps / 2) * (grad_phi + prior_phi) + xi_vec)  # (14)
                    return phi_k

                phi = jax.vmap(_sample_phi_vmap, in_axes=[2, 1, 0])(phi, cwk_t, ck_t)

                return (key, jnp.transpose(phi, (1, 2, 0)), eps), None

            carry = (key, phi, eps)
            (key, phi, _), _ = jax.lax.scan(_sample_phi_loop, carry, (jnp.arange(self.T), CWK, CK))

            # endregion (sample phi)
            # region (sample alpha)
            carry = (key, alpha)
            eta_bars = jnp.array([jnp.mean(eta[t], axis=0) for t in range(self.T)])  # (5)
            (key, alpha), _ = jax.lax.scan(_sample_alpha_loop, carry, (jnp.arange(self.T), eta_bars))
            # endregion (sample alpha)
            model_state = model_state.set(self.index, DTMState(Z, CDK, CWK, CK, alpha, phi, eta, key))
            for t in range(self.T):
                self.diagnosis(t, model_state)
        return model_state

    def diagnosis(self, t: int, model_state: eqx.nn.State) -> None:
        Z, CDK, CWK, CK, alpha, phi, eta, key = model_state.get(self.index)

        N = 0
        total_log_likelihood = 0.0
        softmax_phi = jax.vmap(softmax, in_axes=[1])(phi[t])
        for d in range(self.D[t]):
            N += len(self.W[t][d])
            softmax_eta = softmax(eta[t][d])  # 時刻t文書dのsoftmax
            for n in range(len(self.W[t][d])):  # for each word in document d
                likelihood = softmax_eta @ softmax_phi[:, self.W[t][d][n]]
                if likelihood <= 0:
                    # self.logger.error(f"Likelihood less than 0, error : {likelihood}")
                    total_log_likelihood += -100000000
                    # sys.exit()
                else:
                    total_log_likelihood += jnp.log(likelihood)
        self.logger.info(f"perplexity at time {t+1} : {jax_exp(-total_log_likelihood / N)}")

    # endregion(メソッド)
