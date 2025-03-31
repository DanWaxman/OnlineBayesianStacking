import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, jit, value_and_grad
from functools import partial
import optax
import numpy as np
from jax.scipy.special import logsumexp
from scipy.stats import norm

import argparse

import cvxopt
from cvxopt import matrix, solvers
from tqdm import tqdm

cvxopt.solvers.options["show_progress"] = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SARCOS", help="Dataset name")
parser.add_argument("--N_seeds", type=int, default=10, help="Number of seeds")
parser.add_argument("--compute_ons", type=int, default=1, help="Whether to compute ONS")
parser.add_argument("--exp_grad_learning_rate", type=float, default=1e-2, help="Learning rate for Exponentiated Gradients")
parser.add_argument('--use_fixed_share', type=int, default=0, help='Whether to use a fixed share of the portfolio')

args = parser.parse_args()

compute_ons = args.compute_ons
dataset = args.dataset
N_SEEDS = args.N_seeds
exp_grad_learning_rate = args.exp_grad_learning_rate
use_fixed_share = args.use_fixed_share

# N = np.load("results_garcht_M_10_N_1000_n_5_seed_0.npz")["arr_0"].shape[0]
N = np.load(f"doegp_{dataset}/results_setting_doegp_dataset_{dataset}_seed_0.npz")["arr_0"].shape[0]

# %% Auxiliary functions
def get_weights_expgrad(alpha, pll_t):
    def _step_weights(carry, i):
        log_w = carry

        li = jnp.asarray(pll_t)[:, i - 1]

        # Exponentiated Gradients
        log_w = log_w + alpha * jnp.exp(li - jax.scipy.special.logsumexp(log_w + li))
        log_w = log_w - jax.scipy.special.logsumexp(log_w)

        return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0]) / pll_t.shape[0])
    w = jnp.exp(logw)
    final_log_w, log_ws = lax.scan(
        _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
    )

    # optimized log-weights
    log_ws = jnp.concatenate(
        [logw.reshape(1, -1), log_ws[:-1]], axis=0
    )  # we don't consider the last weight

    return log_ws

if use_fixed_share:
    print("Using fixed share")
    def get_weights_expgrad_fixed_share(alpha, pll_t, delta=1e-2):
        def _step_weights(carry, i):
            log_w = carry

            li = jnp.asarray(pll_t)[:, i - 1]

            # Exponentiated Gradients
            # w_{t+1} = w_t exp(alpha l_t / sum(w l_t))
            log_w = log_w + alpha * jnp.exp(
                li - jax.scipy.special.logsumexp(log_w + li)
            )
            # Project back to simplex with L_1 norm
            log_w = log_w - jax.scipy.special.logsumexp(log_w)

            # Apply fixed share
            log_w = jax.scipy.special.logsumexp(
                jnp.stack(
                    [
                        jnp.log(1 - delta) + log_w,
                        jnp.log(delta)
                        - jnp.ones(log_w.shape[0]) * jnp.log(log_w.shape[0]),
                    ],
                    axis=0,
                ),
                axis=0,
            )
            
            return log_w, log_w

        logw = jnp.log(jnp.ones(pll_t.shape[0]) / pll_t.shape[0])
        w = jnp.exp(logw)
        final_log_w, log_ws = lax.scan(
            _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
        )

        # optimized log-weights
        log_ws = jnp.concatenate(
            [logw.reshape(1, -1), log_ws[:-1]], axis=0
        )  # we don't consider the last weight

        return log_ws

def get_weights_expgrad_bma(alpha, pll_t):
    def _step_weights(carry, i):
        log_w = carry

        li = jnp.asarray(pll_t)[:, i - 1]

        # Exponentiated Gradients
        log_w = log_w + alpha * li
        log_w = log_w - jax.scipy.special.logsumexp(log_w)

        return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0]) / pll_t.shape[0])
    w = jnp.exp(logw)
    final_log_w, log_ws = lax.scan(
        _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
    )

    # optimized log-weights
    log_ws = jnp.concatenate([logw.reshape(1, -1), log_ws[:-1]], axis=0)

    return log_ws


def get_weights_softbayes(alpha, pll_t):
    def _step_weights(carry, i):
        log_w = carry

        li = jnp.asarray(pll_t)[:, i - 1]
        M_t = logsumexp(log_w + li)

        # Soft-Bayes
        log_w = log_w + logsumexp(
            jnp.array([jnp.log1p(-alpha), jnp.log(alpha) + (li - M_t)])
        )

        return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0]) / pll_t.shape[0])
    w = jnp.exp(logw)
    final_log_w, log_ws = lax.scan(
        _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
    )

    # optimized log-weights
    log_ws = jnp.concatenate([logw.reshape(1, -1), log_ws[:-1]], axis=0)

    return log_ws


def get_weights_corrected_softbayes(M, pll_t):
    log_M = jnp.log(M)

    def _step_weights(carry, i):
        alpha_t = jnp.sqrt(log_M / (2 * M * (i)))
        alpha_tp1 = jnp.sqrt(log_M / (2 * M * (i + 1)))

        log_w = carry

        li = jnp.asarray(pll_t)[:, i - 1]
        M_t = logsumexp(log_w + li)

        # Soft-Bayes
        A = log_w
        B = jnp.log(
            1 - alpha_t + alpha_t * jnp.exp(li - M_t)
        )  # logsumexp(jnp.array([jnp.log1p(-alpha_t) * jnp.ones(pll_t.shape[0]), jnp.log(alpha_t) + (li - M_t)]))
        C = jnp.log(alpha_tp1 / alpha_t) * jnp.ones(pll_t.shape[0])
        D = jnp.log(1 - alpha_tp1 / alpha_t) + jnp.ones(pll_t.shape[0]) * jnp.log(1 / M)
        # print(A.shape, B.shape, C.shape, D.shape)
        # print((A + B + C).shape)
        # print(D.shape)
        log_w = logsumexp(jnp.array([A + B + C, D]), axis=0)
        # print(carry.shape, log_w.shape)

        return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0]) / pll_t.shape[0])
    w = jnp.exp(logw)
    final_log_w, log_ws = lax.scan(
        _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
    )

    # optimized log-weights
    log_ws = jnp.concatenate([logw.reshape(1, -1), log_ws[:-1]], axis=0)

    return log_ws


def get_static_weights(pll_t):  # best constant rebalanced portfolio (BCRP)

    def neg_log_wealth(log_weights, pll_t):  # log_weights are not normalized!
        log_mix = logsumexp(log_weights + pll_t.T - logsumexp(log_weights), axis=1)

        return -jnp.sum(log_mix)

    neg_log_wealth_jit = jit(neg_log_wealth)

    init_params = jnp.log(
        jnp.ones(pll_t.shape[0])
        + 0.3 * jax.random.normal(jax.random.PRNGKey(my_seed + 20), (pll_t.shape[0],))
    )

    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(init_params)

    # Define the update step
    @partial(jax.jit)
    def update(params, opt_state, pll_t):
        loss, grads = value_and_grad(neg_log_wealth_jit)(params, pll_t)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # Training loop
    params = init_params
    num_steps = 1000
    loss_vals = []
    for step in range(num_steps):
        loss, params, opt_state = update(params, opt_state, pll_t)
        loss_vals.append(loss)

    static_weights = jnp.exp(params - logsumexp(params))

    return static_weights


# Modified from the Universal Portfolios library
# https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py
# Available under MIT License
class ONS:
    def __init__(self, delta=1 / 8, beta=1e-5, eta=0.0):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super().__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta

    def init_weights(self, m):
        self.A = np.mat(np.eye(m))
        self.b = np.mat(np.zeros(m)).T

        return np.ones(m) / m

    def step(self, r, p):
        # calculate gradient
        grad = np.mat(r / np.dot(p, r)).T
        # update A
        self.A += grad * grad.T
        # update b
        self.b += (1 + 1.0 / self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M):
        """Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
        m = M.shape[0]

        P = matrix(2 * M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A = matrix(np.ones((1, m)))
        b = matrix(1.0)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol["x"])


def get_weights_ons(delta, beta, eta, pll_t):
    reward_t = np.exp(pll_t)
    max_reward_t = reward_t.max(0, keepdims=True)
    reward_t_norm = reward_t / max_reward_t

    ons = ONS(delta, beta, eta)
    current_weights = ons.init_weights(pll_t.shape[0])
    all_weights = [current_weights]

    for current_reward in reward_t_norm.T:
        try:
            current_weights = ons.step(current_reward, current_weights)
            all_weights.append(current_weights)
        except:
            print(current_reward)
            print(current_weights)
            raise Exception("Error in ONS")

    all_weights = np.maximum(
        np.stack(all_weights), np.zeros_like(np.stack(all_weights)) + 1e-64
    )

    return all_weights


# %% Generating the data
# J =

for my_seed in tqdm(range(N_SEEDS)):
    # pll_t = np.load(f"results_garcht_M_10_N_1000_n_5_seed_{my_seed}.npz")["arr_0"].T
    pll_t = np.load(f"doegp_{dataset}/results_setting_doegp_dataset_{dataset}_seed_{my_seed}.npz")[
        "arr_0"
    ].T
    # Get shape of array
    J, N = pll_t.shape

    # Find NaN values
    nan_mask = np.isnan(pll_t)

    # For each row with NaNs, replace with mean of neighboring columns
    for i in range(J):
        nan_idx = np.where(nan_mask[i])[0]
        for idx in nan_idx:
            # Get window of 10 values on each side, excluding NaNs
            window_start = max(0, idx - 10)
            window_end = min(N, idx + 11)
            window = pll_t[i, window_start:window_end]
            window = window[~np.isnan(window)]

            # Replace NaN with mean of window
            if len(window) > 0:
                pll_t[i, idx] = np.mean(window)

    rewards = {}
    weights = {}

    # computing the weights
    logws_eg = get_weights_expgrad(exp_grad_learning_rate, pll_t)
    weights["eg"] = logws_eg

    logws_bma = get_weights_expgrad_bma(1, pll_t)
    weights["bma"] = logws_bma

    static_weights = get_static_weights(pll_t)
    weights["static"] = static_weights

    logws_softbayes = get_weights_corrected_softbayes(J, pll_t)
    weights["softbayes"] = logws_softbayes

    # computing rewards
    reward_t_eg = np.cumsum(logsumexp(pll_t.T + logws_eg, axis=1)) / np.arange(1, N + 1)
    rewards["eg"] = reward_t_eg

    reward_t_bma = np.cumsum(logsumexp(pll_t.T + logws_bma, axis=1)) / np.arange(
        1, N + 1
    )
    rewards["bma"] = reward_t_bma

    reward_t_static = np.cumsum(
        logsumexp(pll_t.T + np.log(static_weights), axis=1)
    ) / np.arange(1, N + 1)
    rewards["static"] = reward_t_static

    reward_t_softbayes = np.cumsum(
        logsumexp(pll_t.T + logws_softbayes, axis=1)
    ) / np.arange(1, N + 1)
    rewards["softbayes"] = reward_t_softbayes

    if compute_ons:
        # normalizing the rewards (exponentiated log predictive values)
        ws_ons = get_weights_ons(0.8, 1e-2, 0.01, pll_t)
        weights["ons"] = ws_ons

        reward_t_ons = np.cumsum(
            logsumexp(pll_t.T + np.log(ws_ons[:-1]), axis=1)
        ) / np.arange(1, N + 1)
        rewards["ons"] = reward_t_ons

        np.savez(
            f"results_doegp_dataset_{dataset}_seed_{my_seed}.npz",
            logws_eg,
            reward_t_eg,
            logws_bma,
            reward_t_bma,
            static_weights,
            reward_t_static,
            ws_ons,
            reward_t_ons,
            logws_softbayes,
            reward_t_softbayes,
        )
    else:
        np.savez(
            f"results_doegp_dataset_{dataset}_seed_{my_seed}_no_ons.npz",
            logws_eg,
            reward_t_eg,
            logws_bma,
            reward_t_bma,
            static_weights,
            reward_t_static,
            logws_softbayes,
            reward_t_softbayes,
        )