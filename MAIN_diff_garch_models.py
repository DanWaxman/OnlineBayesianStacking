import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import yfinance as yf
from scipy.special import gamma, logsumexp
import math
import matplotlib.colors as mcolors
from tqdm import tqdm
import argparse
from garch_models import *



parser = argparse.ArgumentParser(
    prog="subsetreg",
)
parser.add_argument("-M", "--num_models", default=1, type=int)
parser.add_argument("-r", "--rand_seed", default=0, type=int)
parser.add_argument("-N", "--num_par", default=100, type=int)
parser.add_argument("-s", "--mcmc_steps", default=1, type=int)


args = parser.parse_args()

num_models_per_class = args.num_models
seed = args.rand_seed
num_particles = args.num_par
mcmc_steps = args.mcmc_steps

print(f"Running with seed {seed}: {num_models_per_class*4} models, {num_particles} particles, and {mcmc_steps} mcmc steps")


np.random.seed(seed)

#%% Function for loading the S&P
def load_sp500_data(
    symbol: str = "^GSPC",
    start: str = "2015-01-01",
    end: str = "2020-01-01"
) -> pd.DataFrame:
    """
    Uses yfinance to download historical data for the given symbol (default S&P 500),
    flattens any multi-index columns, computes daily percentage returns, and
    returns a DataFrame with a 'returns' column.
    """
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol} in range {start} to {end}.")

    # 1) Flatten multi-level columns if needed
    #    Some yfinance calls may return a multi-index if multiple tickers or certain parameters are used.
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to a single level
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Example: ('Close','^GSPC') => 'Close_^GSPC'
        # This ensures columns become simple strings.

    # 2) Identify the "Close" column name
    #    If you downloaded just one ticker, you might have "Close"
    #    or something like "Close_^GSPC". We'll guess it starts with "Close".
    possible_close_cols = [c for c in df.columns if c.startswith("Close")]
    if len(possible_close_cols) == 1:
        close_col = possible_close_cols[0]
    else:
        raise ValueError(
            f"Could not uniquely identify the 'Close' column. Found: {possible_close_cols}"
        )

    # 3) Compute daily returns based on that close column
    df["returns"] = df[close_col].pct_change()
    df.dropna(subset=["returns"], inplace=True)

    return df


#%% Student's t GARCH(1,1) model
class GARCHtModel:
    """
    GARCH(1,1) with Student-t innovations.
    Particle format: [omega, alpha, beta, nu, sigma_t^2].
    """
    def __init__(
        self,
        # Prior hyperparams for (omega, alpha, beta)
        omega_mean: float = 0.1,
        omega_std: float = 0.1,
        alpha_mean: float = 0.05,
        alpha_std: float = 0.05,
        beta_mean: float = 0.8,
        beta_std: float = 0.1,
        # Prior for degrees of freedom (nu):
        nu_min: float = 3.0,
        nu_max: float = 30.0,
        # Prior for initial sigma^2
        init_sigma2_shape: float = 2.0,
        init_sigma2_scale: float = 2.0
    ):
        """
          omega ~ Normal(omega_mean, omega_std^2), forced >0
          alpha ~ Normal(alpha_mean, alpha_std^2), forced >0
          beta  ~ Normal(beta_mean,  beta_std^2),  forced >0
          nu    ~ Uniform(nu_min, nu_max)
          sigma^2_0 ~ InverseGamma(init_sigma2_shape, init_sigma2_scale)
        """
        self.omega_mean = omega_mean
        self.omega_std = omega_std
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.beta_mean  = beta_mean
        self.beta_std   = beta_std
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.init_sigma2_shape = init_sigma2_shape
        self.init_sigma2_scale = init_sigma2_scale  # the scale of the inverse-gamma is the rate of the gamma distribution from which it's derived

    # -----------------------
    #  A) PRIOR SAMPLING
    # -----------------------
    def sample_prior_params(self, num_particles: int) -> np.ndarray:
        """
        Return (num_particles, 5) => [omega, alpha, beta, nu, sigma_t^2].
        """
        # Sampling from TRUNCATED Gaussian (now it's consistent with function log_prior)
        omega = truncnorm.rvs(-self.omega_mean / self.omega_std,   # a
                              np.inf,                              # b
                              loc=self.omega_mean, 
                              scale=self.omega_std, 
                              size=num_particles)
        alpha = truncnorm.rvs(-self.alpha_mean / self.alpha_std,   # a
                              np.inf,                              # b
                              loc=self.alpha_mean, 
                              scale=self.alpha_std, 
                              size=num_particles)
        beta = truncnorm.rvs(-self.beta_mean / self.beta_std,   # a
                              np.inf,                              # b
                              loc=self.beta_mean, 
                              scale=self.beta_std, 
                              size=num_particles)

        # uniform for nu
        nu = np.random.uniform(self.nu_min, self.nu_max, size=num_particles)
        # inverse gamma for initial sigma^2
        var_init = 1.0 / np.random.gamma(
            shape=self.init_sigma2_shape,
            scale=1/self.init_sigma2_scale,   # we specify the scale as the inverse of the rate; this rate will be the scale of the inverse-gamma
            size=num_particles
        )
        return np.column_stack([omega, alpha, beta, nu, var_init])

    # -----------------------
    #  B) PRIOR DENSITY (LOG)
    # -----------------------
    def log_prior(self, params: np.ndarray) -> np.ndarray:
        """
        Vectorized unnormalized log-prior for param = [omega, alpha, beta, nu, sigma^2].

        params shape: (N, 5)
        Returns: logp shape: (N,)
        """
        # Ensure at least 2D
        params = np.atleast_2d(params)
        
        # Unpack columns
        omega   = params[:, 0]
        alpha   = params[:, 1]
        beta    = params[:, 2]
        nu      = params[:, 3]
        sigma2  = params[:, 4]

        # Make an output array full of -∞
        # We'll overwrite entries with valid priors where constraints are satisfied.
        N = params.shape[0]
        logp = np.full(N, -np.inf, dtype=float)

        # -----------------------------
        # 1) Check constraints
        # -----------------------------
        # All must be positive except nu just has a min/max
        valid_mask = (
            (omega >= 0) 
            & (alpha >= 0) 
            & (beta >= 0) 
            & (sigma2 > 0)
            & (nu >= self.nu_min)
            & (nu <= self.nu_max)
        )

        # Identify valid indices
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            # If none are valid, just return -∞ for all
            return logp
        
        # -----------------------------
        # 2) Compute log prior for valid entries
        # -----------------------------
        # We'll do it in a batch for the subset of valid parameters
        ov  = omega[valid_idx]
        av  = alpha[valid_idx]
        bv  = beta[valid_idx]
        nv  = nu[valid_idx]
        sv2 = sigma2[valid_idx]

        # i) Normal(omega_mean, omega_std^2)
        lp_omega = -0.5 * ((ov - self.omega_mean) / self.omega_std) ** 2
        
        # ii) Normal for alpha, beta
        lp_alpha = -0.5 * ((av - self.alpha_mean) / self.alpha_std) ** 2
        lp_beta  = -0.5 * ((bv - self.beta_mean)  / self.beta_std)  ** 2

        # iii) Uniform for nu in [nu_min, nu_max], which is 0.0 inside, -∞ outside
        # Already enforced by valid_mask, so it contributes 0 here:
        lp_nu = np.zeros_like(nv)

        # iv) Inverse-gamma for sigma^2 => shape = init_sigma2_shape, scale = init_sigma2_scale
        shape = self.init_sigma2_shape
        scale = self.init_sigma2_scale
        # log p(sigma^2) ∝ -(shape+1)*ln(sigma^2) - (scale / sigma^2)
        lp_sigma = -(shape+1)*np.log(sv2) - (scale / sv2)

        # Sum them up
        lp_valid = lp_omega + lp_alpha + lp_beta + lp_nu + lp_sigma

        # -----------------------------
        # 3) Assign results back
        # -----------------------------
        logp[valid_idx] = lp_valid

        return logp


    # -----------------------
    #  C) LIKELIHOOD 
    # -----------------------
    def log_likelihood(self, params: np.ndarray, x_current: float, x_prev: float) -> np.ndarray:
        """
        x_current ~ StudentT(df=nu, loc=0, scale=sqrt(sigma_t^2)).
        Vectorized for all particles in 'params'.
        """
        if len(params.shape) < 2:
            params = params.reshape(1,-1)
       
        omega   = params[:, 0]
        alpha   = params[:, 1]
        beta    = params[:, 2]
        nu      = params[:, 3]
        sigma2_t = params[:, 4]
        sigma_t = np.sqrt(sigma2_t)

        z = x_current / sigma_t
        half_nup1 = 0.5 * (nu + 1.0)

        logC = (
            np.log(gamma(half_nup1))
            - np.log(gamma(0.5*nu))
            - 0.5*np.log(math.pi*nu)
            - np.log(sigma_t)
        )
        log_kernel = -half_nup1 * np.log(1.0 + (z**2)/nu)
        log_pdf = logC + log_kernel
        return log_pdf
    

    # -----------------------
    #  D) STATE TRANSITION
    # -----------------------
    def transition_params(self, params: np.ndarray, x_current: float) -> np.ndarray:
        """
        sigma_{t+1}^2 = omega + alpha*x_t^2 + beta*sigma_t^2
        Keep (omega, alpha, beta, nu) the same, update last col with next sigma^2.
        """
        omega   = params[:, 0]
        alpha   = params[:, 1]
        beta    = params[:, 2]
        nu      = params[:, 3]
        sigma2_t = params[:, 4]

        sigma2_next = omega + alpha*(x_current**2) + beta*sigma2_t #+ 0.1*np.random.rand()  # deterministic recursion, given the particles
        new_params = np.copy(params)
        new_params[:, 4] = sigma2_next
        return new_params
    

    def student_t_logpdf(self,x, nu, scale):
        """
        Vectorized PDF of Student-t(0, scale, df=nu).
        x, nu, scale can be arrays of the same shape. Return same shape result.
        """
        z = x / scale
        half_nup1 = 0.5 * (nu + 1)
        logC = ( 
            np.log(gamma(half_nup1)) 
            - np.log(gamma(0.5*nu)) 
            - 0.5*np.log(math.pi*nu)
            - np.log(scale)
        )
        log_kernel = -half_nup1 * np.log(1 + (z**2)/nu)
        return logC + log_kernel




#%% SMC filter


class SMCFilterGARCH:
    """
    Particle filter for GARCH(1,1)-t with MCMC rejuvenation (optional).
    Each particle: [omega, alpha, beta, nu, sigma_{t}^2].
    """

    def __init__(self, model, num_particles: int, mcmc_steps: int = 0):
        self.model = model
        self.num_particles = num_particles
        self.mcmc_steps = mcmc_steps

        self.params = None
        self.log_weights = None
        self.weights = None

        # Diagonal steps for random-walk MCMC if we do rejuvenation
        self.proposal_scale = np.array([0.01, 0.01, 0.01, 0.5, 0.01])

    def initialize(self):
        self.params = self.model.sample_prior_params(self.num_particles)
        self.log_weights = -np.log(self.num_particles)*np.ones(self.num_particles) 
        self.weights = np.exp(self.log_weights)

    def update(self, x_current: float, x_prev: float):
        """
        1) Evaluate likelihood
        2) Weight update
        3) Systematic resample
        4) GARCH recursion for sigma_{t+1}^2
        5) Optional MCMC rejuvenation
        """
        # 1) Evaluate likelihood p(x_current | current sigma_t^2, etc.)
        log_likelihoods = self.model.log_likelihood(self.params, x_current, x_prev)
        unnormalized = self.log_weights + log_likelihoods

        total = np.exp(logsumexp(unnormalized))
        if total < 1e-20:
            # degenerate => reset
            self.log_weights = -np.log(self.num_particles)*np.ones(self.num_particles) 
            self.weights = np.exp(self.log_weights)
        else:
            self.log_weights = unnormalized - logsumexp(unnormalized)
            self.weights = np.exp(self.log_weights)

        # 2) Systematic resample
        idx = self.systematic_resample()
        self.params = self.params[idx]

        # reset weights after resampling
        self.log_weights = -np.log(self.num_particles)*np.ones(self.num_particles) 
        self.weights = np.exp(self.log_weights)

        # 3) Transition: sigma_t^2 -> sigma_{t+1}^2
        self.params = self.model.transition_params(self.params, x_current)

        # 4) MCMC rejuvenation (if mcmc_steps>0)
        if self.mcmc_steps > 0:
            self.mcmc_rejuvenation(x_current, x_prev)

    def systematic_resample(self):
        """
        Systematic Resampling: returns an array of indices of the same length as weights.
        If weights are shape (N,), the return is shape (N,) with integers in [0..N-1].
        """
        N = len(self.weights)
        # Make a cumulative sum of weights
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0  # guard against floating rounding
        # positions are a regular subdivision of [0,1)
        positions = (np.arange(N) + np.random.rand()) / N

        indices = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cdf[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def mcmc_rejuvenation(self, x_current: float, x_prev: float):
        """
        Vectorized version of MCMC rejuvenation step for all particles at once.
        """
        # Current parameters have shape (num_particles, 5)
        param_old = self.params.copy()  # (num_particles, 5)
        
        # Evaluate the log-posterior for all particles at once
        # log_prior and log_likelihood both return shape (num_particles,)
        logp_old = (
            self.model.log_prior(param_old)
            + self.model.log_likelihood(param_old, x_current, x_prev)
        )  # shape (num_particles,)
        
        # Perform MCMC steps
        for step in range(self.mcmc_steps):
            # Sample proposals for ALL particles at once
            # shape: (num_particles, 5)
            # param_prop = param_old + self.proposal_scale * np.random.randn(*param_old.shape)
            param_prop = param_old + 0.01 * np.random.randn(*param_old.shape)
            
            # Evaluate log posterior for the proposed parameters
            logp_prop = (
                self.model.log_prior(param_prop)
                + self.model.log_likelihood(param_prop, x_current, x_prev)
            )  # shape (num_particles,)

            # Compute acceptance ratio for each particle
            accept_ratio = np.exp(logp_prop - logp_old)  # shape (num_particles,)
            
            # Draw uniform randoms for accept/reject
            uniform_draws = np.random.rand(self.num_particles)  # shape (num_particles,)
            accept_mask = uniform_draws < accept_ratio          # shape (num_particles,)
            
            # Update parameters/logp for those accepted
            param_old[accept_mask] = param_prop[accept_mask]
            logp_old[accept_mask] = logp_prop[accept_mask]

        # Store final samples back
        self.params = param_old



    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights**2)

    # -----------
    # Performance
    # -----------
    def posterior_predictive_logdensity(self, x_current: float, x_prev: float) -> float:
        """
        Compute the posterior predictive log-density for x_current
        given that we have a set of particles 'self.params' representing
        p(params | data up to time t-1).

        This function is now *generic*:
        - We ask the model to do transition_params(...) for x_prev,
        yielding the 'time t' parameters that define p(x_current | ...)
        - Then we ask the model for log_likelihood(...) at x_current.

        Returns a single float = log( average of p(x_current|param_i) ).
        """
        N = self.num_particles

        # 1) Transition old params => new params that generate x_current
        #    (In GARCH terms, we do sigma^2_{t} = some recursion of sigma^2_{t-1}, x_{t-1}.)
        updated_params = self.model.transition_params(self.params, x_prev)
        
        # 2) Evaluate log-likelihood under each updated particle
        logpdf_vals = self.model.log_likelihood(updated_params, x_current, x_prev)
        
        # 3) Posterior predictive density => average in linear space => log-sum-exp
        #   log(1/N * sum(exp(logpdf_vals))) = logsumexp(logpdf_vals) - log(N)
        return logsumexp(logpdf_vals) - np.log(N)



    # def predict_next(self, x_current: float) -> np.ndarray:
    #     """
    #     One-step-ahead predictive sample for x_{t+1}.
    #     sigma_{t+1}^2 = omega + alpha*x_current^2 + beta*sigma_t^2
    #     x_{t+1} ~ StudentT(df=nu, loc=0, scale=sqrt(sigma_{t+1}^2))
    #     """
    #     omega   = self.params[:, 0]
    #     alpha   = self.params[:, 1]
    #     beta    = self.params[:, 2]
    #     nu      = self.params[:, 3]
    #     sigma2_t = self.params[:, 4]

    #     sigma2_next = omega + alpha*(x_current**2) + beta*sigma2_t
    #     sigma_next = np.sqrt(sigma2_next)

    #     # Vector of size num_particles, each with its own scale & dof
    #     z = np.random.standard_t(df=nu)  # shape (num_particles,)
    #     return sigma_next * z



# % ------------------------------------------------------------------------------------------ %

#%% Creating the model instances by sampling prior parameters for each class
def create_diverse_models_per_class(num_models_per_class):
    """
    Create 2 instances for each of the 4 classes => total 8 models.
    We randomly sample hyperparameters from a small uniform range (as an example).
    """
    model_list = []

    # 1) Create 2 GARCHNormalModel
    for _ in range(2):
        omega_mean  = np.random.uniform(0.01, 0.1)
        omega_std   = np.random.uniform(0.01, 0.05)
        alpha_mean  = np.random.uniform(0.01, 0.2)
        alpha_std   = np.random.uniform(0.01, 0.1)
        beta_mean   = np.random.uniform(0.5,  0.95)
        beta_std    = np.random.uniform(0.01, 0.1)
        sigma2_mean = np.random.uniform(0.01, 0.1)
        sigma2_std  = np.random.uniform(0.01, 0.05)

        model = GARCHNormalModel(
            omega_mean=omega_mean,
            omega_std=omega_std,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            beta_mean=beta_mean,
            beta_std=beta_std,
            sigma2_mean=sigma2_mean,
            sigma2_std=sigma2_std
        )
        model_list.append(model)

    # 2) Create 2 GJRGARCHNormalModel
    for _ in range(2):
        omega_mean  = np.random.uniform(0.01, 0.1)
        omega_std   = np.random.uniform(0.01, 0.05)
        alpha_mean  = np.random.uniform(0.01, 0.2)
        alpha_std   = np.random.uniform(0.01, 0.1)
        gamma_mean  = np.random.uniform(0.0,  0.2)
        gamma_std   = np.random.uniform(0.01, 0.05)
        beta_mean   = np.random.uniform(0.5,  0.95)
        beta_std    = np.random.uniform(0.01, 0.1)
        sigma2_mean = np.random.uniform(0.01, 0.1)
        sigma2_std  = np.random.uniform(0.01, 0.05)

        model = GJRGARCHNormalModel(
            omega_mean=omega_mean,
            omega_std=omega_std,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            gamma_mean=gamma_mean,
            gamma_std=gamma_std,
            beta_mean=beta_mean,
            beta_std=beta_std,
            sigma2_mean=sigma2_mean,
            sigma2_std=sigma2_std
        )
        model_list.append(model)

    # 3) Create 2 EGARCHNormalModel
    for _ in range(2):
        omega_mean  = np.random.uniform(-0.5, 0.5)
        omega_std   = np.random.uniform(0.01, 0.2)
        alpha_mean  = np.random.uniform(-0.1, 0.3)
        alpha_std   = np.random.uniform(0.01, 0.1)
        beta_mean   = np.random.uniform(0.5,  0.98)
        beta_std    = np.random.uniform(0.01, 0.1)
        h0_mean     = np.random.uniform(-2.0, 1.0)
        h0_std      = np.random.uniform(0.01, 0.5)
        param_low   = -5.0
        param_high  =  5.0

        model = EGARCHNormalModel(
            omega_mean=omega_mean,
            omega_std=omega_std,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            beta_mean=beta_mean,
            beta_std=beta_std,
            h0_mean=h0_mean,
            h0_std=h0_std,
            param_low=param_low,
            param_high=param_high
        )
        model_list.append(model)

    # 4) Create 2 GARCHtModel (Student-t)
    for _ in range(2):
        omega_mean = np.random.uniform(0.01, 0.1)
        omega_std  = np.random.uniform(0.01, 0.05)
        alpha_mean = np.random.uniform(0.01, 0.2)
        alpha_std  = np.random.uniform(0.01, 0.1)
        beta_mean  = np.random.uniform(0.5, 0.9)
        beta_std   = np.random.uniform(0.01, 0.1)
        nu_min     = 3.0
        nu_max     = np.random.uniform(10, 30)

        model = GARCHtModel(
            omega_mean=omega_mean,
            omega_std=omega_std,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            beta_mean=beta_mean,
            beta_std=beta_std,
            nu_min=nu_min,
            nu_max=nu_max,
            init_sigma2_shape=2.0,
            init_sigma2_scale=2.0
        )
        model_list.append(model)

    return model_list


#%% Main function for ensemble model
def ensemble_example_with_diff_garch_models(x_data, 
                                      num_models_per_class, 
                                      num_particles, 
                                      mcmc_steps,
                                      ):
    # 1) Generate a list of diverse models
    model_list = create_diverse_models_per_class(num_models_per_class)
    num_models = len(model_list)  # should be 8 if we do 2 from each class

    # 2) Create SMC filters for each model
    filters = [
        SMCFilterGARCH(model, num_particles, mcmc_steps=mcmc_steps)
        for model in model_list
    ]
    for f in filters:
        f.initialize()

    T = len(x_data)
    pll_models = []

    # 3) Main loop over time
    for t in tqdm(range(T)):
        x_prev = x_data[t - 1] if t > 0 else 0.0
        x_curr = x_data[t]

        # Evaluate posterior predictive log-density for each model
        log_pms = np.array([
            f.posterior_predictive_logdensity(x_curr, x_prev) 
            for f in filters
        ])
        pll_models.append(log_pms)

        # Update each model with the new observation
        for f in filters:
            f.update(x_curr, x_prev)

    return {
        "pll_models": np.stack(pll_models),  # shape (T, num_models)
    }



#%% Load the data 
df = load_sp500_data(symbol="^GSPC", start="2015-01-01", end="2020-01-01")
x_data = df["returns"].values
T = len(x_data)


#%% Run the code

results = ensemble_example_with_diff_garch_models(x_data, 
                                            num_models_per_class=num_models_per_class, 
                                            num_particles=num_particles,
                                            mcmc_steps=mcmc_steps,
                                            ) 



#%% Save the results
np.savez(f"results_diff_garch_models_M_{num_models_per_class*4}_N_{num_particles}_n_{mcmc_steps}_seed_{seed}.npz",
         results["pll_models"])