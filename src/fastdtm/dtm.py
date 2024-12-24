import logging
import sys

import numpy as np
from matplotlib.pylab import f
from scipy.special import softmax
from utilpy import StringBuilder

from configs import ModelConfig

from .alias_table import AliasTable

ZERO = 1e-6


def np_exp(x) -> np.ndarray:
    return np.clip(np.exp(np.clip(x, a_max=700, a_min=-700)), a_max=None, a_min=ZERO)


class DTM:
    def __init__(self, data: list[list[list[int]]], dictionary: list[str], config: ModelConfig):
        self.W: list[list[list[int]]] = data
        """data (time, document at t, wotd index)"""
        self.vocabulary: list[str] = dictionary
        """list of word in data."""
        self.logger: logging.Logger = logging.getLogger(str(__class__))
        self.K: int = config.num_topic
        """num of topic"""
        self.V: int = len(self.vocabulary)
        """num of words"""
        self.T: int = len(self.W)
        """num of time"""
        self.D = np.zeros(self.T, dtype=int)
        """num of document at each time (time, document)"""
        self.sgld_a: float = config.SGLD_a
        self.sgld_b: float = config.SGLD_b
        self.sgld_c: float = config.SGLD_c
        self.dtm_phi_var: float = config.phi_var
        """variance of phi."""
        self.dtm_eta_var: float = config.eta_var
        """variance of eta."""
        self.dtm_alpha_var: float = config.alpha_var
        """variance of alpha."""
        self.Z = []
        """topic assignment. (time, document at t, word)"""
        self.CDK = []
        """counter. (time, num of documnt at t, topic)"""
        self.CWK = np.zeros((self.T, self.V, self.K), dtype=int)
        """counter. (T, V, K)"""
        self.CK = np.zeros((self.T, self.K), dtype=int)
        """counter.(time, topic)"""
        self.alpha = np.zeros((self.T, self.K), dtype=float)
        """time evolving parameters topic latent strengh. (time, topic)"""
        self.phi = np.zeros((self.T, self.V, self.K), dtype=float)
        """time evolving word strength for each topic. (time, word, topic)"""
        self.eta = []
        """topic strengh with uncertainity. (time, num of document at t, topic)"""
        self.term_alias_samples = np.zeros((self.T, self.V, self.K), dtype=int)
        """buffer of random value created from alias table. (time, word, topic)"""

        for t in range(self.T):
            self.D[t] = len(self.W[t])
            self.eta.append(np.zeros((self.D[t], self.K)))
            self.CDK.append(np.zeros((self.D[t], self.K), dtype=int))
            self.Z.append([])
            for d in range(self.D[t]):
                self.Z[t].append(np.zeros(len(self.W[t][d]), dtype=int))
        self.logger.info(f"{self.K=} {self.T=} {self.V=} {self.D=}")

    def initialize(self, init_with_lda: bool):
        init_alpha = 50.0 / self.K
        init_beta = 0.01
        for t in range(self.T):
            for d in range(self.D[t]):
                N = len(self.W[t][d])
                for n in range(N):
                    w = self.W[t][d][n]
                    k = np.random.randint(low=0, high=self.K - 1)  # 初期トピック
                    self.Z[t][d][n] = k
                    self.CDK[t][d][k] += 1
                    self.CWK[t][w][k] += 1
                    self.CK[t][k] += 1
                    self.eta[t][d][k] += (1 + init_alpha) / (N + self.K * init_alpha)
        if init_with_lda:  # t=0のみLDA
            for iter in range(50):
                for d in range(self.D[0]):
                    N = len(self.W[0][d])
                    for n in range(N):
                        k = self.Z[0][d][n]
                        w = self.W[0][d][n]
                        self.CDK[0][d][k] -= 1
                        self.CWK[0][w][k] -= 1
                        self.CK[0][k] -= 1
                        prob = [
                            self.CDK[0][d][k]
                            + init_alpha * (self.CWK[0][w][k] + init_beta) / (self.CK[0][k] + self.V * init_beta)
                            for k in range(self.K)
                        ]

                        k = np.random.choice(self.K, p=prob / np.sum(prob))
                        self.Z[0][d][n] = k
                        self.CDK[0][d][k] += 1
                        self.CWK[0][w][k] += 1
                        self.CK[0][k] += 1
        for t in range(self.T):
            for w in range(self.V):
                for k in range(self.K):
                    self.phi[t][w][k] = (self.CWK[0][w][k] + init_beta) / (self.CK[0][k] + self.V * init_beta)
                self.term_alias_samples[t][w] = self.build_alias_table(t, w)

    def build_alias_table(self, t: int, w: int):
        table = AliasTable()
        table.build(self.phi[t][w])
        return table.sample(self.K)

    def estimate(self, num_iters: int):
        sample_indices = np.zeros((self.T, self.V), dtype=int)
        for iter in range(num_iters):
            self.logger.info(f"epoche {iter+1}")
            eps = self.sgld_a * pow(self.sgld_b + iter, -self.sgld_c)
            mean = np.zeros(self.K)
            for t in range(self.T):
                xi_vec = np.ones(self.K) * np.random.normal(0.0, eps)
                for d in range(self.D[t]):
                    N = len(self.W[t][d])  # num of words at current doc
                    # region (sample eta)
                    prior_eta = (self.alpha[t] - self.eta[t][d]) / self.dtm_eta_var
                    grad_eta = self.CDK[t][d] - N * softmax(self.eta[t][d])  # (10)
                    self.eta[t][d] += (eps / 2) * (grad_eta + prior_eta) + xi_vec  # (10)
                    # endregion (sample eta)
                    # region (sample topic)
                    for n in range(N):
                        for mh in range(4):
                            k = self.Z[t][d][n]
                            w = self.W[t][d][n]
                            self.CDK[t][d][k] -= 1
                            self.CWK[t][w][k] -= 1
                            self.CK[t][k] -= 1

                            if mh % 2 == 0:  # Z-proposal
                                index = np.random.randint(0, N)  # sample random word
                                proposal = self.Z[t][d][index]
                                acceptance_prob = np_exp(self.phi[t][w, proposal]) / np_exp(self.phi[t][w, k])
                            else:
                                if sample_indices[t][w] >= self.K:  # all sampled
                                    self.term_alias_samples[t][w] = self.build_alias_table(t, w)  # rebuild alias table
                                    sample_indices[t][w] = 0
                                proposal = self.term_alias_samples[t][w][sample_indices[t][w].item()]  # sampletd topic
                                sample_indices[t][w] += 1
                                acceptance_prob = np_exp(self.eta[t][d][proposal]) / np_exp(self.eta[t][d][k])
                            acceptance_prob = 1.0 if acceptance_prob > 1.0 else acceptance_prob
                            # metropolis hasting test.
                            if np.random.uniform() >= acceptance_prob:  # reject proposal
                                proposal = k  # ramain old topic
                            self.Z[t][d][n] = proposal
                            self.CDK[t][d][proposal] += 1
                            self.CWK[t][w][proposal] += 1
                            self.CK[t][proposal] += 1
                    # endregion (sample topic)

                # region (sample phi)
                xi_vec = np.ones(self.V) * np.random.normal(0.0, eps)
                for k in range(self.K):
                    if t == 0:
                        phi_sigma = 1.0 / ((1.0 / 100) + (1 / self.dtm_phi_var))
                        prior_phi = self.phi[t + 1][:, k] * (phi_sigma / self.dtm_phi_var)
                        prior_phi = ((2 * prior_phi) - 2 * self.phi[t][:, k]) / self.dtm_phi_var
                    elif t == self.T - 1:
                        prior_phi = (self.phi[t - 1][:, k] - self.phi[t][:, k]) / self.dtm_phi_var
                    else:
                        prior_phi = (
                            self.phi[t + 1][:, k] + self.phi[t - 1][:, k] - 2 * self.phi[t][:, k]
                        ) / self.dtm_phi_var

                    denom_phi = self.CK[t][k] * softmax(self.phi[t][:, k])  # (14) # TODO これでよい？
                    grad_phi = self.CWK[t][:, k] - denom_phi  # (14)
                    self.phi[t][:, k] += (eps / 2) * (grad_phi + prior_phi) + xi_vec  # (14)
                # endregion (sample phi)
                # region (sample alpha)
                alpha_bar = np.zeros(self.K)
                alpha_precision = 0.0  # designed to be a diagonal matrix
                if t == 0:
                    alpha_precision = (1.0 / 100) + (1 / self.dtm_alpha_var)
                    alpha_sigma = 1.0 / alpha_precision
                    alpha_bar = self.alpha[t + 1] * (alpha_sigma / self.dtm_alpha_var)
                elif t == self.T - 1:
                    alpha_bar = (self.alpha[t - 1] - self.alpha[t]) / self.dtm_alpha_var
                    alpha_precision = 1.0 / self.dtm_alpha_var
                else:
                    alpha_precision = 2 / self.dtm_alpha_var
                    alpha_bar = (self.alpha[t + 1] + self.alpha[t - 1]) / 2  # (5)
                eta_bar = self.eta[t].mean(axis=0)  # (5)
                sigma_inv = 1.0 / (alpha_precision + self.D[t] / self.dtm_eta_var)  # (5)
                cov = np.eye(self.K) * sigma_inv  # (5)
                # MEMO: 本実装に誤りがあったため修正
                mean = (
                    alpha_bar + eta_bar - cov @ (eta_bar * alpha_precision + alpha_bar * self.D[t] / self.dtm_eta_var)
                )  # (4)
                self.alpha[t] = np.random.multivariate_normal(mean, cov)  # sample
                # endregion (sample alpha)
                self.diagnosis(t)

    def diagnosis(self, t: int):
        N = 0
        total_log_likelihood = 0.0
        softmax_phi = np.zeros((self.K, self.V))
        for k in range(self.K):
            softmax_phi[k] = softmax(self.phi[t][:, k])
        for d in range(self.D[t]):
            N += len(self.W[t][d])
            softmax_eta = softmax(self.eta[t][d])  # 時刻t文書dのsoftmax
            for n in range(len(self.W[t][d])):  # for each word in document d
                likelihood = softmax_eta @ softmax_phi[:, self.W[t][d][n]]
                if likelihood <= 0:
                    # self.logger.error(f"Likelihood less than 0, error : {likelihood}")
                    total_log_likelihood += -100000000
                    # sys.exit()
                else:
                    total_log_likelihood += np.log(likelihood)
        self.logger.info(f"perplexity at time {t+1} : {np_exp(-total_log_likelihood / N)}")

    def save_data(self, dir: str):
        for t in range(self.T):
            fpath = f"{dir}/time_slice_{t}.txt"
            with open(fpath, mode="w") as f:
                for k in range(self.K):
                    ranking_idx = np.argsort([-self.phi[t][v][k] for v in range(self.V)])
                    f.write(f"Topic {k}\n")
                    for v in range(np.min(self.V, 30)):
                        w = ranking_idx[v]
                        f.write(f"({self.vocabulary[w]}, {self.phi[t][w][k]})\n")
