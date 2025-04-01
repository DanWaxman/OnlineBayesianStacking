#%%
import jax
jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
# from jax import lax, jit, value_and_grad
# from functools import partial
# import optax 
import numpy as np
# from jax.scipy.special import logsumexp
from scipy.stats import norm

from DOEBE.doebe import DOEBE
from DOEBE.models import DOGP

from experiment_utils import get_data

import os

import argparse

# import cvxopt
# from cvxopt import matrix, solvers
# cvxopt.solvers.options['show_progress'] = False

parser = argparse.ArgumentParser(
    prog="subsetreg",
)
parser.add_argument("-N", "--num_data", default=1000000000000000, type=int)
parser.add_argument("-n", "--num_pretrain", default=1000, type=int)
parser.add_argument("-r", "--rand_seed", default=0, type=int)
# parser.add_argument("-o", "--compute_ons", default=1, type=int)
parser.add_argument("-s", "--setting", default='doegp', type=str)
parser.add_argument('-d', "--dataset", default="SARCOS", type=str)


args = parser.parse_args()

N = args.num_data
my_seed = args.rand_seed
n_pre = args.num_pretrain
# compute_ons = args.compute_ons
setting = args.setting
dataset = args.dataset

print(f"Running setting {setting} with seed {my_seed} and (N,n) = ({N},{n_pre})")


#%% 

import objax
objax.random.DEFAULT_GENERATOR.seed(my_seed)

X, y = get_data(dataset)
N = min(N, X.shape[0])
X = X[:N]
y = y[:N]
print("Input shapes:", X.shape, y.shape)
d = X.shape[1]
M = 100 // d
ls_guess = (jnp.max(X, axis=0) - jnp.min(X, axis=0))
sigma_rws = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
n_features = 100
doegp = DOEBE(
    [
        DOGP(
            n_features // 2,
            "rbf",
            d,
            ls_guess * jnp.ones(d),
            1.0,
            sigma_rw,
            0.25,
            train_lengthscale=True,
        )
        for sigma_rw in sigma_rws
    ]
)

doegp.pretrain(X[:n_pre],y[:n_pre])
_, _, ws, yhat, cov_yhat = doegp.fit(X[n_pre:],y[n_pre:],return_ws=True)

# computing the pll_t
J = len(sigma_rws)
pll_t = np.empty((J,N - n_pre))
for j in range(J):
    pll_t[j,:] = norm.logpdf(y[n_pre:],yhat[:,j],np.sqrt(cov_yhat[:,j]))


pll_t = pll_t.T  # shape (T, num_models) where T is the length of the dataset (minus pretraining) and num_models is the number of models    


save_path = f'./{setting}_{dataset}/results_setting_{setting}_dataset_{dataset}_seed_{my_seed}.npz'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

np.savez(save_path, pll_t)



