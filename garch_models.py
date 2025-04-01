import numpy as np
import math
from scipy.stats import truncnorm
from scipy.special import logsumexp


def rvs_truncnorm(mean, std, low, high, size):
    """
    Sample 'size' values from a Normal(mean, std^2) truncated to [low, high].
    """
    a = (low - mean)/std
    b = (high - mean)/std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def logpdf_truncnorm(x, mean, std, low, high):
   # Domain check
    invalid = (x < low) | (x > high)
    # Plain Normal logpdf ignoring the partition constant
    # = -0.5*((x - mean)/std)^2 - 0.5*log(2*pi) - log(std)
    # We'll do it vectorized:
    logpdf = -0.5*((x - mean)/std)**2 - 0.5*np.log(2*np.pi) - np.log(std)
    # If out of domain => -inf
    logpdf[invalid] = -np.inf
    return logpdf





class GARCHNormalModel:
    """
    GARCH(1,1) with Normal(0, sigma_t^2) innovations.

    Parameter layout (4,):
      [omega, alpha, beta, sigma^2].
    """

    def __init__(self,
                 omega_mean=0.1, omega_std=0.05,
                 alpha_mean=0.05, alpha_std=0.05,
                 beta_mean=0.8,  beta_std=0.05,
                 sigma2_mean=0.05, sigma2_std=0.05):
        """
        All these hyperparams define truncated Normal( mean, std^2 ) on [0, ∞).
        """
        self.omega_mean  = omega_mean
        self.omega_std   = omega_std
        self.alpha_mean  = alpha_mean
        self.alpha_std   = alpha_std
        self.beta_mean   = beta_mean
        self.beta_std    = beta_std
        self.sigma2_mean = sigma2_mean
        self.sigma2_std  = sigma2_std

    def sample_prior_params(self, num_particles: int) -> np.ndarray:
        """
        Return shape (num_particles, 4).
        """
        # Draw from truncated normals on [0, ∞)
        omega  = rvs_truncnorm(self.omega_mean,  self.omega_std,  0.0, np.inf, size=num_particles)
        alpha  = rvs_truncnorm(self.alpha_mean,  self.alpha_std,  0.0, np.inf, size=num_particles)
        beta   = rvs_truncnorm(self.beta_mean,   self.beta_std,   0.0, np.inf, size=num_particles)
        sigma2 = rvs_truncnorm(self.sigma2_mean, self.sigma2_std, 0.0, np.inf, size=num_particles)

        return np.column_stack([omega, alpha, beta, sigma2])

    def log_prior(self, params: np.ndarray) -> np.ndarray:
        """
        Vectorized log-prior. params shape = (N,4).
        We'll do truncated Normal for each parameter on [0, ∞).
        """
        omega  = params[:, 0]
        alpha  = params[:, 1]
        beta   = params[:, 2]
        sigma2 = params[:, 3]

        # If any param < 0 => invalid => -inf
        invalid = (omega <= 0) | (alpha <= 0) | (beta <= 0) | (sigma2 <= 0)

        lp_omega  = logpdf_truncnorm(omega,  self.omega_mean,  self.omega_std,  0.0, np.inf)
        lp_alpha  = logpdf_truncnorm(alpha,  self.alpha_mean,  self.alpha_std,  0.0, np.inf)
        lp_beta   = logpdf_truncnorm(beta,   self.beta_mean,   self.beta_std,   0.0, np.inf)
        lp_sigma2 = logpdf_truncnorm(sigma2, self.sigma2_mean, self.sigma2_std, 0.0, np.inf)

        lp = lp_omega + lp_alpha + lp_beta + lp_sigma2
        lp[invalid] = -np.inf
        return lp

    def log_likelihood(self, params: np.ndarray, x_current: float, x_prev: float) -> np.ndarray:
        """
        Vectorized logpdf of N(0, sigma^2_t).
        params shape = (N,4).
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)
        sigma2_t = params[:, 3]
        sigma_t  = np.sqrt(sigma2_t)

        # Gaussian log-pdf:
        #  -0.5 * log(2π) - log(sigma) - 0.5 * (x^2 / sigma^2)
        const_term = -0.5 * np.log(2.0 * math.pi)
        logpdf = const_term - np.log(sigma_t) - 0.5 * (x_current**2 / sigma2_t)
        return logpdf

    def transition_params(self, params: np.ndarray, x_current: float) -> np.ndarray:
        """
        GARCH(1,1) recursion:
          sigma_{t+1}^2 = omega + alpha*x_t^2 + beta*sigma_t^2
        """
        newp = np.copy(params)
        omega   = params[:, 0]
        alpha   = params[:, 1]
        beta    = params[:, 2]
        sigma2_t= params[:, 3]

        sigma2_next = omega + alpha*(x_current**2) + beta*sigma2_t
        newp[:, 3] = sigma2_next
        return newp








class GJRGARCHNormalModel:
    """
    GJR-GARCH(1,1) with Normal innovations.

    Parameter layout (5,):
      [omega, alpha, gamma, beta, sigma2].
    """

    def __init__(self,
                 omega_mean=0.1,  omega_std=0.05,
                 alpha_mean=0.05, alpha_std=0.05,
                 gamma_mean=0.05, gamma_std=0.05,
                 beta_mean=0.8,   beta_std=0.05,
                 sigma2_mean=0.05, sigma2_std=0.05):
        self.omega_mean  = omega_mean
        self.omega_std   = omega_std
        self.alpha_mean  = alpha_mean
        self.alpha_std   = alpha_std
        self.gamma_mean  = gamma_mean
        self.gamma_std   = gamma_std
        self.beta_mean   = beta_mean
        self.beta_std    = beta_std
        self.sigma2_mean = sigma2_mean
        self.sigma2_std  = sigma2_std

    def sample_prior_params(self, num_particles: int) -> np.ndarray:
        """
        Return shape (num_particles, 5).
        """
        omega  = rvs_truncnorm(self.omega_mean,  self.omega_std,  0.0, np.inf, size=num_particles)
        alpha  = rvs_truncnorm(self.alpha_mean,  self.alpha_std,  0.0, np.inf, size=num_particles)
        gamma_ = rvs_truncnorm(self.gamma_mean,  self.gamma_std,  0.0, np.inf, size=num_particles)
        beta   = rvs_truncnorm(self.beta_mean,   self.beta_std,   0.0, np.inf, size=num_particles)
        sigma2 = rvs_truncnorm(self.sigma2_mean, self.sigma2_std, 0.0, np.inf, size=num_particles)

        return np.column_stack([omega, alpha, gamma_, beta, sigma2])

    def log_prior(self, params: np.ndarray) -> np.ndarray:
        """
        Vectorized truncated Normal on [0,∞).
        params shape = (N,5).
        """
        omega  = params[:, 0]
        alpha  = params[:, 1]
        gamma_ = params[:, 2]
        beta   = params[:, 3]
        sigma2 = params[:, 4]

        invalid = (omega <= 0) | (alpha <= 0) | (gamma_ < 0) | (beta <= 0) | (sigma2 <= 0)

        lp_omega  = logpdf_truncnorm(omega,  self.omega_mean,  self.omega_std,  0.0, np.inf)
        lp_alpha  = logpdf_truncnorm(alpha,  self.alpha_mean,  self.alpha_std,  0.0, np.inf)
        lp_gamma  = logpdf_truncnorm(gamma_, self.gamma_mean,  self.gamma_std,  0.0, np.inf)
        lp_beta   = logpdf_truncnorm(beta,   self.beta_mean,   self.beta_std,   0.0, np.inf)
        lp_sigma2 = logpdf_truncnorm(sigma2, self.sigma2_mean, self.sigma2_std, 0.0, np.inf)

        lp = lp_omega + lp_alpha + lp_gamma + lp_beta + lp_sigma2
        lp[invalid] = -np.inf
        return lp

    def log_likelihood(self, params: np.ndarray, x_current: float, x_prev: float) -> np.ndarray:
        """
        Vectorized normal(0, sigma^2_t) logpdf.
        params shape = (N,5).
        """
        if params.ndim==1:
            params = params.reshape(1,-1)
        sigma2_t = params[:,4]
        sigma_t  = np.sqrt(sigma2_t)

        const_term = -0.5 * np.log(2.0*math.pi)
        logpdf = const_term - np.log(sigma_t) - 0.5*(x_current**2 / sigma2_t)
        return logpdf

    def transition_params(self, params: np.ndarray, x_current: float) -> np.ndarray:
        """
        GJR recursion:
          sigma_{t+1}^2 = omega + alpha*x_t^2 + gamma*x_t^2*1_{x_t<0} + beta*sigma_t^2
        """
        newp = np.copy(params)
        omega   = params[:,0]
        alpha   = params[:,1]
        gamma_  = params[:,2]
        beta    = params[:,3]
        sigma2_t= params[:,4]

        indicator = 1.0*(x_current < 0)
        sigma2_next = omega + alpha*(x_current**2) + gamma_*(x_current**2)*indicator + beta*sigma2_t
        newp[:,4] = sigma2_next
        return newp





class EGARCHNormalModel:
    """
    EGARCH(1,1) with Normal(0, exp(h_t)) innovations.
    Parameter layout (4,):
       [omega, alpha, beta, h_t].

    Recursion:
      h_{t+1} = omega + beta*h_t + alpha*(|z_t| - E|z_t|)
      sigma_t^2 = exp(h_t).
    """

    def __init__(self,
                 # We'll allow truncated normal in some wide range for all param
                 omega_mean=0.0, omega_std=0.5,  # can be negative
                 alpha_mean=0.1, alpha_std=0.1,  # can be pos/neg
                 beta_mean=0.8,  beta_std=0.1,   # typically near 1
                 h0_mean=0.0,    h0_std=0.5,
                 # We'll define the truncation bounds for everything:
                 param_low=-5.0, param_high=5.0):
        self.omega_mean = omega_mean
        self.omega_std  = omega_std
        self.alpha_mean = alpha_mean
        self.alpha_std  = alpha_std
        self.beta_mean  = beta_mean
        self.beta_std   = beta_std
        self.h0_mean    = h0_mean
        self.h0_std     = h0_std
        self.low  = param_low
        self.high = param_high

    def sample_prior_params(self, num_particles: int) -> np.ndarray:
        """
        Return shape (num_particles,4).
        Each parameter is truncated Normal(mean, std^2) on [low, high].
        """
        omega = rvs_truncnorm(self.omega_mean, self.omega_std, self.low, self.high, size=num_particles)
        alpha = rvs_truncnorm(self.alpha_mean, self.alpha_std, self.low, self.high, size=num_particles)
        beta  = rvs_truncnorm(self.beta_mean,  self.beta_std,  self.low, self.high, size=num_particles)
        h0    = rvs_truncnorm(self.h0_mean,    self.h0_std,    self.low, self.high, size=num_particles)

        return np.column_stack([omega, alpha, beta, h0])

    def log_prior(self, params: np.ndarray) -> np.ndarray:
        """
        Vectorized truncated Normal on [low, high].
        params shape = (N,4).
        """
        omega = params[:,0]
        alpha = params[:,1]
        beta  = params[:,2]
        h0    = params[:,3]

        # If outside [low, high], we set -inf
        invalid = (omega<self.low)|(omega>self.high)|(alpha<self.low)|(alpha>self.high)\
                  |(beta<self.low)|(beta>self.high)|(h0<self.low)|(h0>self.high)

        lp_omega = logpdf_truncnorm(omega, self.omega_mean, self.omega_std, self.low, self.high)
        lp_alpha = logpdf_truncnorm(alpha, self.alpha_mean, self.alpha_std, self.low, self.high)
        lp_beta  = logpdf_truncnorm(beta,  self.beta_mean,  self.beta_std,  self.low, self.high)
        lp_h0    = logpdf_truncnorm(h0,    self.h0_mean,    self.h0_std,    self.low, self.high)

        lp = lp_omega + lp_alpha + lp_beta + lp_h0
        lp[invalid] = -np.inf
        return lp

    def log_likelihood(self, params: np.ndarray, x_current: float, x_prev: float) -> np.ndarray:
        """
        Vectorized Normal(0, exp(h_t)) log-pdf.
        params shape = (N,4).
        """
        if params.ndim==1:
            params = params.reshape(1, -1)
        h_t = params[:,3]
        sigma2_t = np.exp(h_t)
        sigma_t  = np.sqrt(sigma2_t)

        const_term = -0.5* np.log(2.0*math.pi)
        logpdf = const_term - np.log(sigma_t) - 0.5*(x_current**2 / sigma2_t)
        return logpdf

    def transition_params(self, params: np.ndarray, x_current: float) -> np.ndarray:
        """
        Minimal EGARCH recursion ignoring asymmetry:
          h_{t+1} = omega + beta*h_t + alpha*(|z_t| - E|z_t|)
        We'll approximate E|z_t| for a Normal(0,1) by sqrt(2/pi).
        """
        newp = np.copy(params)
        omega = params[:,0]
        alpha = params[:,1]
        beta  = params[:,2]
        h_t   = params[:,3]

        sigma2_t = np.exp(h_t)
        sigma_t  = np.sqrt(sigma2_t)
        z_t      = x_current / sigma_t

        # For Normal(0,1), E|Z| = sqrt(2/pi).
        c = math.sqrt(2.0/math.pi)
        h_next = omega + beta*h_t + alpha*(np.abs(z_t) - c)
        newp[:,3] = h_next
        return newp
