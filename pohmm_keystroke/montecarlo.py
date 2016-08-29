import numpy as np
import pandas as pd
from scipy.integrate import quad, trapz
from statsmodels.distributions import ECDF

from .io import load_data, load_results, save_results
from .data import preprocess_data, DATASETS
from .plotting import plot_cdf, plot_pvalues, save_fig
from .classify import pohmm_factory


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    See http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def area_test_statistic(model_cdf_fn, empirical_cdf_fn):
    """
    Area between the model and empirical CDF

    See R. Malmgren, "A Poissonian explanation for heavy tails in e-mail communication"
    """
    fn = lambda x: np.abs(model_cdf_fn(x) - empirical_cdf_fn(x))
    # Split the integral up into the increasing piecewise function and constant (past the max empirical value)
    upper = empirical_cdf_fn.x.max()
    # x = np.linspace(0, upper, 10000)
    # a1 = trapz(fn(x), x)
    a1 = trapz(savitzky_golay(fn(np.linspace(0, upper, 1000)), 101, 2), np.linspace(0, upper, 1000))
    a2, _ = quad(fn, upper, np.inf)
    area = a1 + a2
    return area


def montecarlo(df, fit_fn, col, m):
    """
    Test the goodness of fit of a model to a dataset
    """
    # Find the best fit parameters for the empirical data
    best_fit_model = fit_fn(df)

    emp_data = df[col].values

    # Compute the test statistic between the fitted model
    model_test_stat = area_test_statistic(best_fit_model.cdf_fn(feature=col), ECDF(emp_data))

    # Accumulate enough test statistics
    surrogate_test_stats = np.zeros(m)
    seeds = np.random.randint(0, 1e8, size=m)
    for i, seed in enumerate(seeds):
        surrogate_data = pd.concat(
            [best_fit_model.sample_df(pstates=x['event'], random_state=seed) for _, x in df.groupby(level=[0, 1])])
        surrogate_fit_model = fit_fn(surrogate_data)
        surrogate_test_stats[i] = area_test_statistic(surrogate_fit_model.cdf_fn(feature=col),
                                                      ECDF(surrogate_data[col].values))

    # Biased p value
    pvalue = (1 + (np.abs(surrogate_test_stats - surrogate_test_stats.mean()) > np.abs(
        model_test_stat - surrogate_test_stats.mean())).sum()) / (m + 1)
    print(df.index.get_level_values(0)[0], ',', pvalue, flush=True)
    return pvalue


def montecarlo_pvalues(df, fit_fn, col, m):
    """
    Test the null hypthosesis: model is consistent with the data
    """
    pvalues = df.groupby(level=[0]).apply(lambda x: montecarlo(x, fit_fn, col, m))
    pvalues = pd.DataFrame(pvalues, columns=['pvalue'])
    return pvalues


def dataset_goodness_of_fit(dataset, event, m=100, threshold=0.05, out_name=None, seed=1234):
    np.random.seed(1234)
    print('Running:', out_name, flush=True)

    df = load_data(dataset)

    if dataset == 'password':
        df = df[df.index.get_level_values(1).isin(np.arange(150, 200))]

    df = preprocess_data(df, event, ['tau'])

    # Discard first tau which was set to the median during feature extraction
    df = df.groupby(level=[0, 1]).apply(lambda x: x[1:]).reset_index(level=[1, 2], drop=True)

    # Select some random users to plot CDF/PDF
    users = sorted(np.unique(df.index.get_level_values(0)))
    for i in range(2):
        plot_cdf(df[df.index.get_level_values(0) == users[i]], pohmm_factory, 'tau')
        save_fig(out_name + '_user%d_CDF' % (i + 1))

    # goodness of fit test
    # pvalues = load_results(out_name + '_pvalues')
    pvalues = montecarlo_pvalues(df, pohmm_factory, 'tau', m=m)
    save_results(pvalues, out_name + '_pvalues')

    # Plot the distribution of pvalues
    plot_pvalues(pvalues, threshold)
    save_fig(out_name)

    # Summary
    n_models = len(pvalues)
    n_rejected = (pvalues <= threshold).sum()
    print(dataset, 'Rejected %d out of %d models at %.2f%% (%.2f rejection rate)'
          % (n_rejected, n_models, threshold, n_rejected / n_models), flush=True)


def goodness_of_fit_results():

    for dataset in DATASETS:
        dataset_goodness_of_fit(dataset, 'keyname', out_name='%s_pohmm_montecarlo' % dataset)
        dataset_goodness_of_fit(dataset, 'none', out_name='%s_hmm_montecarlo' % dataset)
