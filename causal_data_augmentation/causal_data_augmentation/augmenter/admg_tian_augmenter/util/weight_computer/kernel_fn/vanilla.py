from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.nonparametric import kernels
from statsmodels.sandbox.nonparametric import kernels as sandbox_kernels
from statsmodels.nonparametric import bandwidths

import sys
# MINIMUM_CONTI_BANDWIDTH = sys.float_info.min
MINIMUM_CONTI_BANDWIDTH = 1e-100


def indicator_kernel(h: np.ndarray, Xi: np.ndarray, x: np.ndarray):
    """
    Params:
        h : not used. This argument is left for compatibility.
        Xi : 1-D ndarray, shape (nobs, K). The value of the training set.
        x : 1-D ndarray, shape (K, 1). The value at which the kernel density is being estimated.

    Returns:
        kernel_value : ndarray, shape (n_obs, K).
                       The value of the kernel function at each training point for each var.
    """
    return (Xi - x) == 0


def epanechnikov(h: np.ndarray, Xi: np.ndarray, x: np.ndarray):
    """Epanechnikov kernel.
    Params:
        h : bandwidth.
        Xi : 1-D ndarray, shape (nobs, 1). The value of the training set.
        x : 1-D ndarray, shape (1, nbatch). The value at which the kernel density is being estimated.

    Returns:
        kernel_value : ndarray, shape (n_obs, nbatch).
                       The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    out = 3 / 4 * (1 - u**2) * (np.abs(u) <= 1)
    assert out.shape == (Xi.shape[0], x.shape[1])
    return out


def triweight(h: np.ndarray, Xi: np.ndarray, x: np.ndarray):
    """Triweight kernel.

    Params:
        h : bandwidth.
        Xi : 1-D ndarray, shape (nobs, 1). The value of the training set.
        x : 1-D ndarray, shape (1, nbatch). The value at which the kernel density is being estimated.

    Returns:
        kernel_value : ndarray, shape (n_obs, nbatch).
                       The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    out = 35 / 32 * (np.maximum(0, 1 - u**2)**3)
    assert out.shape == (Xi.shape[0], x.shape[1])
    return out


def biweight(h: np.ndarray, Xi: np.ndarray, x: np.ndarray):
    """Biweight kernel.

    Params:
        h : bandwidth.
        Xi : 1-D ndarray, shape (nobs, 1). The value of the training set.
        x : 1-D ndarray, shape (1, nbatch). The value at which the kernel density is being estimated.

    Returns:
        kernel_value : ndarray, shape (n_obs, nbatch).
                       The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    out = 15 / 16 * (np.maximum(0, 1 - u**2)**2)
    assert out.shape == (Xi.shape[0], x.shape[1])
    return out


def tricube(h: np.ndarray, Xi: np.ndarray, x: np.ndarray):
    """Tricube kernel.

    Params:
        h : bandwidth.
        Xi : 1-D ndarray, shape (nobs, 1). The value of the training set.
        x : 1-D ndarray, shape (1, nbatch). The value at which the kernel density is being estimated.

    Returns:
        kernel_value : ndarray, shape (n_obs, nbatch).
                       The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    out = 70 / 81 * (np.maximum(0, 1 - np.abs(u)**3)**3)
    assert out.shape == (Xi.shape[0], x.shape[1])
    return out


# https://github.com/statsmodels/statsmodels/blob/2a5a6ec3baf901f52008aee10f166ff6085d3ba5/statsmodels/nonparametric/_kernel_base.py
# statsmodels.nonparametric.kernels: https://github.com/statsmodels/statsmodels/blob/2a5a6ec3baf901f52008aee10f166ff6085d3ba5/statsmodels/nonparametric/kernels.py
kernel_func = dict(
    wangryzin=kernels.wang_ryzin,
    aitchisonaitken=kernels.aitchison_aitken,

    # https://tedboy.github.io/statsmodels_doc/_modules/statsmodels/nonparametric/kernels.html#gaussian
    gaussian=kernels.gaussian,
    aitchison_aitken_reg=kernels.aitchison_aitken_reg,
    wangryzin_reg=kernels.wang_ryzin_reg,
    gauss_convolution=kernels.gaussian_convolution,
    wangryzin_convolution=kernels.wang_ryzin_convolution,
    aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
    gaussian_cdf=kernels.gaussian_cdf,
    aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
    wangryzin_cdf=kernels.wang_ryzin_cdf,
    d_gaussian=kernels.d_gaussian,
    # tricube=kernels.tricube,
    tricube=tricube,
    # Following are added here:
    indicator=indicator_kernel,
    epanechnikov=epanechnikov,
    triweight=triweight,
    biweight=biweight,
)


@dataclass
class VanillaProductKernelConfig:
    """A configuration set used for product kernels.

    Params:
        conti_kertype : Default: 'gaussian'.
        ordered_kertype : statmodels' original default is 'wangryzin'.
        unordered_kertype : statmodels' original default is 'aitchisonaitken'.
        conti_bw_method :
        ordered_bw_method :
        unordered_bw_method :
    """
    conti_kertype: str = 'gaussian'
    conti_bw_method: str = 'normal_reference'
    conti_bw_temperature: float = 1.

    ordered_kertype: str = 'indicator'
    ordered_bw_method: str = 'indicator'

    unordered_kertype: str = 'indicator'
    unordered_bw_method: str = 'indicator'


def bw_normal_reference(x, kernel=sandbox_kernels.Gaussian):
    """
    Plug-in bandwidth with kernel specific constant based on normal reference.

    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Used to calculate the constant for the plug-in bandwidth.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    C = kernel().normal_reference_constant
    A = bandwidths._select_sigma(x)
    n = len(x)
    return C * A * n**(-0.2)


class BandwidthNormalReference:
    def __init__(self, coeff=1):
        self.coeff = coeff

    def __call__(self, *args, **kwargs):
        return self.coeff * bw_normal_reference(*args, **kwargs)


class VanillaProductKernel:
    """Product kernel object.

    Notes:
      Bandwidth methods: ``statsmodels.nonparametric.bandwidths``: https://www.statsmodels.org/devel/_modules/statsmodels/nonparametric/bandwidths.html
    """

    BW_METHODS = {
        'normal_reference': BandwidthNormalReference(),
        'indicator': lambda x: None,
    }

    def __init__(
            self,
            data_ref: np.ndarray,
            vartypes: str,
            config: VanillaProductKernelConfig = VanillaProductKernelConfig()):
        """Constructor.

        Params:
            data_ref : Reference data points for which the kernel values are computed.
            vartypes : The variable type ('c': continuous, 'o': ordered, 'u': unordered). Example: ``'ccou'``.
            product_kernel_config : the configuration object.
        """
        self.vartypes = vartypes
        self.kertypes = dict(c=config.conti_kertype,
                             o=config.ordered_kertype,
                             u=config.unordered_kertype)
        self.bw_methods = dict(c=config.conti_bw_method,
                               o=config.ordered_bw_method,
                               u=config.unordered_bw_method)
        self.conti_bw_temperature = config.conti_bw_temperature
        self._fit(data_ref)

    def _fit(self, data_ref):
        """
        Params:
            data_ref : ndarray of shape ``(n_obs, n_dim)`` the kernel centers.
        """
        self.data_ref = data_ref
        self.bandwidths = []
        for k, vtype in enumerate(self.vartypes):
            bw_method = self.bw_methods.get(vtype, lambda x: 'not implemented')
            if isinstance(bw_method, str):
                bw = self.BW_METHODS[bw_method](data_ref[:, k])
            else:
                bw = bw_method(data_ref[:, k])

            if vtype == 'c':
                bw = bw * self.conti_bw_temperature
                if bw == 0:
                    # Error handling for the case that there is only one unique value for the variable in the data.
                    bw = MINIMUM_CONTI_BANDWIDTH
            self.bandwidths.append(bw)

    def __call__(self, data):
        """Compute the kernel matrix ``(k(data_i, data_ref_j))_{ij}``.

        Parameters:
            data : ndarray of shape ``(n_data, n_dim)``.

        Returns: ndarray of shape ``(n_data, n_data_ref)``
        """
        gram_matrices = []
        for k, vtype in enumerate(self.vartypes):
            func = kernel_func[self.kertypes[vtype]]
            gram_matrix = func(self.bandwidths[k], data[:, k][:, None],
                               self.data_ref[:, k][None, :])
            gram_matrices.append(gram_matrix)
        # assert np.isnan(np.array(gram_matrices).prod(axis=0)).sum() == 0
        return np.array(gram_matrices).prod(axis=0)
