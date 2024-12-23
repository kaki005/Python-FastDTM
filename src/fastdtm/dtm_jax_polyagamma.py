import logging
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import softmax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar

from .util_func import jax_exp, softmax_multi_posterior


class DTMState(NamedTuple):
    flatZ: Int[Array, "all_word"]
    """topic assignment. (word)"""
    flatCDK: Float[Array, "document K"]
    """counter. (document, topic)"""
    CWK: Float[Array, "T V K"]
    """counter. (T, V, K)"""
    CK: Float[Array, "T K"]
    """counter.(time, topic)"""
    alpha: Float[Array, "T K"]
    """time evolving parameters topic latent strengh. (time, topic)"""
    phi: Float[Array, "T V K"]
    """time evolving word strength for each topic. (time, word, topic)"""
    flat_eta: Float[Array, "document K"]
    """topic strengh with uncertainity. (document topic)"""
    key: PRNGKeyArray
    # term_alias_samples = np.zeros((self.T, self.V, self.K), dtype=int)
    # """buffer of random value created from alias table. (time, word, topic)"""


class DTM_PolyaGamma(eqx.Module):
    # ===========================
    # region (property)
    # ===========================
    logger: logging.Logger = eqx.field(static=True)
    W: list[list[Float[Array, "N_dt"]]] = eqx.field(static=True)
    """data (time, document at t, wotd index)"""
    flatW: Float[Array, "all_word"]
    doc_indexes: Int[Array, "all_word"]
    """index of document for each word."""
    time_ind_per_word: Int[Array, "all_word"]
    """index of time for each word."""
    time_ind_per_doc: Int[Array, "all_doc"]
    """index of time for each document."""
    Ns: list[Float[Array, "D_t"]] = eqx.field(static=True)
    """num of words for each document. (time, document)"""
    N_per_word: Float[Array, "all_word"]
    flatNs: Float[Array, "all_doc"]
    vocabulary: list[str] = eqx.field(static=True)
    """list of word in data."""
    K: int = eqx.field(static=True)
    """num of topic"""
    V: int = eqx.field(static=True)
    """num of words"""
    T: int = eqx.field(static=True)
    """num of time"""
    D: Int[Array, "T"]
    """num of document at each time (time, document)"""
    all_word_num: int = eqx.field(static=True)
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
        self.logger = logging.getLogger(str(__class__))
        self.W = [[jnp.array(w_td) for w_td in w_t] for w_t in W]
        self.Ns = [jnp.array([len(w_td) for w_td in w_t]) for w_t in W]
        self.vocabulary = dictionary
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
        doc_indexes = []
        time_ind_per_word = []
        flatW = []
        N_per_word = []
        flatNs = []
        time_ind_per_doc = []
        doc_index = 0
        for t in range(self.T):
            for d in range(self.D[t]):
                time_ind_per_word += [t] * self.W[t][d].shape[0]
                doc_indexes += [doc_index] * self.W[t][d].shape[0]
                N_per_word += [self.Ns[t][d]] * self.W[t][d].shape[0]
                flatW += W[t][d]
                flatNs.append(self.Ns[t][d])
                doc_index += 1
            time_ind_per_doc += [t] * int(self.D[t].item())
        self.doc_indexes = jnp.array(doc_indexes)
        self.time_ind_per_word = jnp.array(time_ind_per_word)
        self.flatW = jnp.array(flatW)
        self.all_word_num = self.flatW.shape[0]
        self.N_per_word = jnp.array(N_per_word)
        self.flatNs = jnp.array(flatNs)
        self.time_ind_per_doc = jnp.array(time_ind_per_doc)

        # 初期化
        key = random.key(config.seed)
        key, subkey = random.split(key)
        flatZ = random.randint(subkey, (self.all_word_num,), 0, self.K)
        flatCDK = jnp.zeros((jnp.sum(self.D), self.K), dtype=int)
        CWK = jnp.zeros((self.T, self.V, self.K), dtype=int)
        CK = jnp.zeros((self.T, self.K), dtype=int)
        key, subkey = random.split(key)
        alpha = random.normal(subkey, (self.T, self.K), dtype=float)
        key, subkey = random.split(key)
        phi = random.normal(subkey, (self.T, self.V, self.K), dtype=float)
        flat_eta = jnp.zeros((jnp.sum(self.D), self.K), dtype=float)
        self.index = eqx.nn.StateIndex(DTMState(flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, key))
        # region(log)
        self.logger.info(f"{self.T=}")
        self.logger.info(f"{self.V=}")
        self.logger.info(f"{self.D=}")
        self.logger.info(f"{self.all_word_num=}")
        self.logger.info(f"{self.K=}")
        # endregion(log)

    def initialize(self, model_state: eqx.nn.State, init_with_lda=True) -> eqx.nn.State:
        init_alpha = 50.0 / self.K
        init_beta = 0.01

        def _add_topic_loop(carry, params):
            flatCDK, CWK, CK, flat_eta, key = carry
            t, d, w, N, k = params
            # key, subkey = random.split(key)
            # k = random.randint(subkey, (1,), 0, self.K - 1)
            flatCDK = flatCDK.at[d, k].add(1)
            CWK = CWK.at[t, w, k].add(1)
            CK = CK.at[t, k].add(1)
            flat_eta = flat_eta.at[d, k].add((1 + init_alpha) / (N + self.K * init_alpha))
            return (flatCDK, CWK, CK, flat_eta, key), None

        flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, key = model_state.get(self.index)
        carry = (flatCDK, CWK, CK, flat_eta, key)
        params = (
            self.time_ind_per_word,
            self.doc_indexes,
            self.flatW,
            self.N_per_word,
            flatZ,
        )
        # sample topic for each word
        (flatCDK, CWK, CK, flat_eta, key), _ = jax.lax.scan(_add_topic_loop, carry, params)
        if init_with_lda:  # t=0のみLDA
            doc_indexes = self.doc_indexes[self.time_ind_per_word == 0]
            word_indexes = self.flatW[self.time_ind_per_word == 0]

            def _lda_loop(carry, params):
                flatCDK, CWK, CK = carry
                d, w, pre_topic, key = params
                flatCDK = flatCDK.at[d, pre_topic].subtract(1)
                CWK = CWK.at[0, w, pre_topic].subtract(1)
                CK = CK.at[0, pre_topic].subtract(1)
                prob = jax.vmap(
                    lambda k: flatCDK[d, k] + init_alpha * (CWK[0, w, k] + init_beta) / (CK[0, k] + self.V * init_beta)
                )(jnp.arange(self.K))
                key, subkey = random.split(key)
                new_topic = random.categorical(subkey, jnp.log(prob))
                flatCDK = flatCDK.at[d, new_topic].add(1)
                CWK = CWK.at[0, w, new_topic].add(1)
                CK = CK.at[0, new_topic].add(1)
                return (flatCDK, CWK, CK), new_topic

            for iter in range(50):
                carry = (flatCDK, CWK, CK)
                key, subkey = random.split(key)

                keys = random.split(subkey, doc_indexes.shape[0])
                pre_topics = flatZ[self.time_ind_per_word == 0]
                params = (doc_indexes, word_indexes, pre_topics, keys)
                (flatCDK, CWK, CK), new_flatZ = jax.lax.scan(_lda_loop, carry, params)
                flatZ = flatZ.at[self.time_ind_per_word == 0].set(new_flatZ)
        new_state = model_state.set(self.index, DTMState(flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, key))
        return new_state

    # endregion(コンストラクタ,初期化)

    # ===========================
    # region (メソッド)
    # ===========================
    def estimate(self, model_state: eqx.nn.State, num_iters: int) -> eqx.nn.State:
        @eqx.filter_jit
        def _sample_eta(
            key: PRNGKeyArray,
            eta_td: Float[Array, "K"],
            CDK_td: Float[Array, "K"],
            alpha_t: Float[Array, "K"],
        ):
            (key, subkey) = random.split(key)
            mean, cov = softmax_multi_posterior(eta_td, self.dtm_eta_var * jnp.eye(self.K), CDK_td, key)
            eta_td = random.multivariate_normal(subkey, mean + alpha_t, cov, dtype=float)
            return eta_td

        @eqx.filter_jit
        def _sample_topic_loop(carry, params):
            flatCDK, CWK, CK, flat_eta, phi = carry
            t, d, w, key, pre_topic = params
            eta_td = flat_eta[d]

            def _mh_test_word(key: PRNGKeyArray, pre_topic: Int[Scalar, "1"]):
                key, subkey = random.split(key)
                # index = random.randint(subkey, (1,), 0, N).squeeze()  # sample random word
                # proposal = Z_t[index]  # その単語のトピック
                proposal = random.randint(subkey, (1,), 0, self.K).squeeze()  # sample random toipc
                acceptance_prob = jax_exp(phi[t, w, proposal]) / jax_exp(phi[t, w, pre_topic])
                return proposal, acceptance_prob, key

            def _mh_test_topic(key: PRNGKeyArray, pre_topic: Int[Scalar, "1"]):
                key, subkey = random.split(key)
                proposal = random.randint(subkey, (1,), 0, self.K).squeeze()  # sample random toipc
                acceptance_prob = jax_exp(eta_td[proposal]) / jax_exp(eta_td[pre_topic])
                return proposal, acceptance_prob, key

            def _sample(key: PRNGKeyArray, is_mh_word: bool, pre_topic: Int[Scalar, "1"]):
                # metropolis hasting test.
                proposal, acceptance_prob, key = jax.lax.cond(is_mh_word, _mh_test_word, _mh_test_topic, key, pre_topic)
                key, subkey = random.split(key)
                is_rejected = random.uniform(subkey) >= acceptance_prob
                new_topic = is_rejected * pre_topic + (1 - is_rejected) * proposal
                return new_topic

            CWK = CWK.at[t, w, pre_topic].subtract(1)
            CK = CK.at[t, pre_topic].subtract(1)
            flatCDK = flatCDK.at[d, pre_topic].subtract(1)
            keys = random.split(key, 3)
            new_topic = _sample(keys[0], True, pre_topic)
            new_topic = _sample(keys[1], False, new_topic)
            CWK = CWK.at[t, w, new_topic].add(1)
            CK = CK.at[t, new_topic].add(1)
            flatCDK = flatCDK.at[d, new_topic].add(1)
            return (flatCDK, CWK, CK, flat_eta, phi), new_topic

        @eqx.filter_jit
        def _sample_phi_tk(
            t: float,
            phi_tk: Float[Array, "V"],
            phi_tk_plus,
            phi_tk_minus,
            cw_tk: Float[Array, "V"],
            c_tk,
            key: PRNGKeyArray,
        ):
            def _left():
                precision = (1.0 / 100) + (1 / self.dtm_phi_var)
                sigma = 1.0 / precision
                phi_tk_bar = phi_tk_plus * (sigma / self.dtm_phi_var)
                return phi_tk_bar

            def _right():
                phi_tk_bar = (phi_tk_minus - phi_tk) / self.dtm_alpha_var
                # precision = 1.0 / self.dtm_alpha_var
                return phi_tk_bar

            def _middle():
                return (phi_tk_plus + phi_tk_minus) / 2  # (5)

            key, subkey = random.split(key)
            phi_tk_bar = jnp.select([t == 0, t == self.T - 1], [_left(), _right()], default=_middle())
            mean, cov = softmax_multi_posterior(phi_tk, self.dtm_phi_var / 2 * jnp.eye(self.V), cw_tk, subkey)
            return random.multivariate_normal(key, mean + phi_tk_bar, cov)

        @eqx.filter_jit
        def _sample_phit(
            t,
            phi_t: Float[Array, "V K"],
            phi_t_plus: Float[Array, "V K"],
            phi_t_minus: Float[Array, "V K"],
            cwk_t: Int[Array, "V K"],
            ck_t: Int[Array, "K"],
            key: PRNGKeyArray,
        ) -> Float[Array, "V K"]:
            keys = random.split(key, self.K)
            phi_t = jax.vmap(_sample_phi_tk, in_axes=[None, 1, 1, 1, 1, 0, 0])(
                t, phi_t, phi_t_plus, phi_t_minus, cwk_t, ck_t, keys
            )
            return phi_t.T

        @eqx.filter_jit
        def _sample_alpha(
            t,
            alpha_t: Float[Array, "V K"],
            alpha_t_plus: Float[Array, "V K"],
            alpha_t_minus: Float[Array, "V K"],
            key: PRNGKeyArray,
            eta_bar: Float[Array, "K"],
        ) -> Float[Array, "K"]:
            def _left():
                precision = (1.0 / 100) + (1 / self.dtm_alpha_var)
                alpha_sigma = 1.0 / precision
                alpha_bar = alpha_t_plus * (alpha_sigma / self.dtm_alpha_var)
                return alpha_bar

            def _right():
                alpha_bar = (alpha_t_minus - alpha_t) / self.dtm_alpha_var
                # precision = 1.0 / self.dtm_alpha_var
                return alpha_bar

            def _middle():
                # precision = 2 / self.dtm_alpha_var
                alpha_bar = (alpha_t_plus + alpha_t_minus) / 2  # (5)
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
            new_alpha_t = random.multivariate_normal(key, mean, cov)
            return new_alpha_t

        for i in range(num_iters):
            self.logger.info(f"epoche {i+1}")
            flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, key = model_state.get(self.index)
            eps = self.sgld_a * jnp.pow(self.sgld_b + i, -self.sgld_c).squeeze()
            key, subkey, subkey2 = random.split(key, 3)
            xi = random.normal(subkey2, (1,)).squeeze() * eps
            # region (sample eta)
            alpha_per_doc = []
            keys = random.split(subkey, (jnp.sum(self.D),))
            for t in range(self.T):
                alpha_per_doc += [alpha[t]] * int(self.D[t].item())
            flat_eta = jax.vmap(_sample_eta)(keys, flat_eta, flatCDK, jnp.array(alpha_per_doc))
            # endregion (sample eta)

            # region (sample topic)
            key, subkey = random.split(key)
            keys = random.split(subkey, self.all_word_num)
            carry = (flatCDK, CWK, CK, flat_eta, phi)
            params = (self.time_ind_per_word, self.doc_indexes, self.flatW, keys, flatZ)
            (flatCDK, CWK, CK, flat_eta, _), flatZ = jax.lax.scan(_sample_topic_loop, carry, params)
            # endregion (sample topic)

            # region (sample phi)
            phi_plus = jnp.concat([phi[1:], jnp.zeros((1, self.V, self.K))], axis=0)
            key, subkey = random.split(key)
            keys = random.split(subkey, self.T)
            phi_minus = jnp.concat([jnp.zeros((1, self.V, self.K)), phi[:-1]], axis=0)
            phi = jax.vmap(_sample_phit, in_axes=[0, 0, 0, 0, 0, 0, 0])(
                jnp.arange(self.T), phi, phi_plus, phi_minus, CWK, CK, keys
            )

            # endregion (sample phi)

            # region (sample alpha)
            key, subkey = random.split(key)
            keys = random.split(subkey, self.T)
            eta_bar = jnp.array([jnp.mean(flat_eta[self.time_ind_per_doc == t]) for t in range(self.T)])  # (5)
            alpha_plus = jnp.concat([alpha[1:], jnp.zeros((1, self.K))])
            alpha_minus = jnp.concat([jnp.zeros((1, self.K)), alpha[:-1]])
            alpha = jax.vmap(_sample_alpha, in_axes=[0, 0, 0, 0, 0, 0])(
                jnp.arange(self.T), alpha, alpha_plus, alpha_minus, keys, eta_bar
            )
            # endregion (sample alpha)
            model_state = model_state.set(self.index, DTMState(flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, key))
            self.diagnosis(model_state)
        return model_state

    def diagnosis(self, model_state: eqx.nn.State) -> None:
        flatZ, flatCDK, CWK, CK, alpha, phi, flat_eta, _ = model_state.get(self.index)
        self.logger.info(f"{flatZ=}")

        def _log_likelihood(total_llh, params):
            t, d, w = params
            softmax_phi = softmax(phi[t][w])
            softmax_eta = softmax(flat_eta[d])  # 時刻t文書dのsoftmax
            likelihood = softmax_eta @ softmax_phi
            is_negative = likelihood <= 0
            llh = is_negative * jnp.inf + (1 - is_negative) * jnp.log(likelihood)
            return total_llh + llh, None

        total_log_likelihood, _ = jax.lax.scan(
            _log_likelihood, 0.0, (self.time_ind_per_word, self.doc_indexes, self.flatW)
        )
        self.logger.info(f"perplexity : {jax_exp(-total_log_likelihood / self.all_word_num)}")

    # endregion(メソッド)
