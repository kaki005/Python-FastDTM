import logging

import numpy as np
from matplotlib.pylab import f
from scipy.special import softmax
from utilpy import StringBuilder

from configs import ModelConfig

from .alias_table import AliasTable


class DTM:
    def __init__(self, data: list[list[list[int]]], dictionary: list[str], config: ModelConfig):
        self.W: list[list[list[int]]] = data
        """data (time, num of document at t, num of word)"""
        self.vocabulary: list[str] = dictionary
        self.K: int = config.num_topic
        self.sgld_a: float = config.SGLD_a
        self.sgld_b: float = config.SGLD_b
        self.sgld_c: float = config.SGLD_c
        self.dtm_phi_var: float = config.phi_var
        self.dtm_eta_var: float = config.eta_var
        self.dtm_alpha_var: float = config.alpha_var
        self.V: int = len(self.vocabulary)
        """num of words"""
        self.T: int = len(self.W)
        """num of time"""
        self.D = np.zeros(self.T, dtype=int)
        """num of document at each time (time, document)"""
        self.Z = []
        self.CDK = []
        self.CWK = np.zeros((self.T, self.V, self.K), dtype=int)
        """(time, word, topic)"""
        self.CK = np.zeros((self.T, self.K), dtype=int)
        """(time, topic)"""
        self.alpha = np.zeros((self.T, self.K), dtype=float)
        """(time, topic)"""
        self.phi = np.zeros((self.T, self.V, self.K), dtype=float)
        """(time, word, topic)"""
        self.eta = []
        """(time, num of document at t, topic)"""
        self.logger: logging.Logger = logging.getLogger(str(__class__))
        self.term_alias_samples = np.zeros((self.T, self.V, self.K), dtype=int)

        for t in range(self.T):
            self.D[t] = len(self.W[t])
            self.eta.append(np.zeros((self.D[t], self.K)))

            self.CDK.append(np.zeros((self.D[t], self.K), dtype=int))
            self.Z.append([])
            for d in range(self.D[t]):
                self.Z[t].append(np.zeros(len(self.W[t][d]), dtype=int))

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
                    self.CDK[t][d][n] += 1
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
                        self.CDK[0][d][n] -= 1
                        self.CWK[0][w][k] -= 1
                        self.CK[0][k] -= 1
                        prob = [
                            self.CDK[0][d][k]
                            + init_alpha * (self.CWK[0][w][k] + init_beta) / (self.CK[0][k] + self.V * init_beta)
                            for k in range(self.K)
                        ]

                        k = np.random.choice(self.K, p=prob / np.sum(prob))
                        self.Z[0][d][n] = k
                        self.CDK[0][d][n] += 1
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
            self.logger.info(f"iter: {iter}")
            eps = self.sgld_a * pow(self.sgld_b + iter, -self.sgld_c)
            xi = np.random.normal(0.0, eps)
            mean = np.zeros(self.K)
            for t in range(self.T):
                xi_vec = np.ones(self.K) * xi
                for d in range(self.D[t]):
                    N = len(self.W[t][d])
                    # estimate eta
                    soft_eta = softmax(self.eta[t][d])
                    prior_eta = (self.alpha[t] - self.eta[t][d]) / self.dtm_eta_var
                    denom_eta = N * soft_eta
                    grad_eta = self.CDK[t][d].transpose() - denom_eta
                    self.eta[t][d] += ((eps / 2) * (grad_eta + prior_eta)) + xi_vec
                    for n in range(N):
                        for mh in range(4):
                            k = self.Z[t][d][n]
                            w = self.W[t][d][n]
                            self.CDK[t][d][k] -= 1
                            self.CWK[t][w][k] -= 1
                            self.CK[t][k] -= 1

                            if mh % 2 == 0:
                                # Z-proposal
                                index = np.random.randint(0, N)
                                proposal = self.Z[t][d][index]
                                acceptance_prob = np.exp(self.phi[t][w, proposal]) / np.exp(self.phi[t][w, k])
                            else:
                                if sample_indices[t][w] >= self.K:
                                    self.term_alias_samples[t][w] = self.build_alias_table(t, w)
                                    sample_indices[t][w] = 0
                                proposal = self.term_alias_samples[t][w][sample_indices[t][w].item()]
                                sample_indices[t][w] += 1
                                acceptance_prob = np.exp(self.eta[t][d][proposal]) / np.exp(self.eta[t][d][k])
                            # acceptance_prob = 1.0 if acceptance_prob > 1.0 else acceptance_prob

                            if np.random.uniform() >= acceptance_prob:
                                # reject proposal
                                proposal = k
                                self.Z[t][d][n] = k
                                self.CDK[t][d][n] += 1
                                self.CWK[t][w][k] += 1
                                self.CK[t][k] += 1

                xi_vec = np.ones(self.V) * xi
                for k in range(self.K):
                    # sample phi
                    soft_phi = softmax(self.phi[t][:, k])  # TODO これでよい？
                    prior_phi = np.zeros(self.V)
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

                    denom_phi = self.CK[t][k] * soft_phi
                    grad_phi = self.CWK[t][:, k] - denom_phi
                    self.phi[t][:, k] += ((eps / 2) * (grad_phi + prior_phi)) + xi_vec

                # sample alpha
                alpha_bar = np.zeros(self.K)
                alpha_precision = 0.0  # designed to be a diagonal matrix
                cov = np.eye(self.K)
                if t == 0:
                    alpha_precision = (1.0 / 100) + (1 / self.dtm_alpha_var)
                    alpha_sigma = 1.0 / alpha_precision
                    alpha_bar = self.alpha[t + 1] * (alpha_sigma / self.dtm_alpha_var)
                elif t == self.T - 1:
                    alpha_bar = (self.alpha[t - 1] - self.alpha[t]) / self.dtm_alpha_var
                    alpha_precision = 1.0 / self.dtm_alpha_var
                else:
                    alpha_precision = 2 / self.dtm_alpha_var
                    alpha_bar = (self.alpha[t + 1] - self.alpha[t - 1]) / 2
                    eta_bar = self.eta[t].sum(axis=1)
                    sigma = 1.0 / (1.0 / alpha_precision + (self.D[t] / self.dtm_eta_var))
                    cov *= sigma
                    mean = (alpha_bar / alpha_precision + (eta_bar / self.dtm_eta_var)) * sigma
                    self.alpha[t] = np.random.multivariate_normal(mean, cov)
                self.diagnosis(t)

    def diagnosis(self, t: int):
        N = 0
        total_log_likelihood = 0.0
        softmax_phi = np.zeros((self.K, self.V))
        softmax_eta = np.zeros((self.D[t], self.K))
        for k in range(self.K):
            softmax_phi[k] = softmax(self.phi[t][:, k])
        for d in range(self.D[t]):
            N += len(self.W[t][d])
            softmax_eta[d] = softmax(self.eta[t][d])
            for n in range(len(self.W[t][d])):
                likelihood = 0.0
                w = self.W[t][d][n]
                for k in range(self.K):
                    likelihood += softmax_eta[d][k] * softmax_phi[k][w]
                    if likelihood < 0:
                        self.logger.error("Likelihood less than 0, error")
                total_log_likelihood += np.log(likelihood)
        self.logger.info(f"perplexity at {t} : {np.exp(-total_log_likelihood / N)}")

    def save_data(self, dir: str):
        for t in range(self.T):
            fpath = f"{dir}/time_slice_{t}.txt"
            with open(fpath, mode="w") as f:
                for k in range(self.K):
                    ranking_idx = np.argsort([-self.phi[t][v][k] for v in range(self.V)])
                    f.write(f"Topic {k}\n")
                    for v in range(self.V):
                        w = ranking_idx[v]
                        f.write(f"({self.vocabulary[w]}, {self.phi[t][w][k]})\n")