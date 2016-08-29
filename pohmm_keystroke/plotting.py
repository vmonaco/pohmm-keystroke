import os
import numpy as np
import pandas as pd
from scipy import interp
from statsmodels.distributions import ECDF

import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from . import FIGURES_DIR

sns.set(context='notebook', font_scale=3.0, font='sans-serif')
sns.set_palette(sns.color_palette('Set1', n_colors=5)[::-1])

POHMM_COLOR = sns.xkcd_rgb['denim blue']
HMM_COLOR = sns.xkcd_rgb['reddish brown']


def save_fig(name, ext='pdf'):
    plt.savefig(os.path.join(FIGURES_DIR, name + '.%s' % ext), bbox_inches='tight')
    plt.close()
    return


def plot_error(roc):
    """
    Plot far and frr as a function of threshold
    """
    roc = roc.copy()

    min_thresh, max_thresh = roc['threshold'].min(), roc['threshold'].max()

    far = roc[['fold', 'threshold', 'far']]
    far.columns = ['fold', 'threshold', 'value']
    far['Error'] = 'FAR'

    frr = roc[['fold', 'threshold', 'frr']]
    frr.columns = ['fold', 'threshold', 'value']
    frr['Error'] = 'FRR'

    roc = pd.DataFrame(pd.concat([frr, far]))

    sns.set(style='darkgrid')
    sns.set_context('notebook', font_scale=3.0)
    plt.figure(figsize=(8, 6))

    g = sns.tsplot(roc, time='threshold', unit='fold', condition='Error', value='value', color='cubehelix', ci=95)
    g.set_xlabel('Threshold')
    g.set_ylabel('Error')
    g.set_xlim(min_thresh, max_thresh)
    g.set_ylim(0, 1)
    plt.legend(title=None, loc='lower left')
    return


def plot_roc(roc, condition, pivot):
    """
    Plot an roc curve with confidence bands
    """
    # Interpolate to get tpr for each fpr
    far = np.linspace(0, 1, 1000)

    def _interp(df):
        df = pd.DataFrame({'far': far, 'frr': interp(far[::-1], df['far'][::-1], df['frr'][::-1])[::-1]})
        return df

    if type(roc) == list:
        rocs = []
        for name, r in roc:
            r = r.groupby(pivot).apply(_interp).reset_index(level=1, drop=True).reset_index()
            r[condition] = name
            rocs.append(r)
        roc = pd.concat(rocs)
    else:
        roc = roc.groupby(pivot).apply(_interp).reset_index(level=1, drop=True).reset_index()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    g = sns.tsplot(roc, time='far', unit=pivot, condition=condition, value='frr', ci=95)
    g.set_xlabel('False acceptance rate')
    g.set_ylabel('False rejection rate')
    g.set_xlim(0, 1)
    g.set_ylim(0, 1)
    plt.legend(title=None)
    return


def plot_penalty_example(penalty):
    """
    Show the penalty function for genuine and impostor users
    """
    genuine_idx = penalty['reference_user'] == penalty['query_user']
    genuine = penalty[genuine_idx]
    impostor = penalty[~genuine_idx]
    thresh = genuine['penalty'].max()
    impostor.loc[:, 'type'] = 'Impostor'

    sns.set(style='darkgrid')
    sns.set_context('notebook', font_scale=3.0)
    plt.figure(figsize=(16, 10))

    g = sns.tsplot(impostor, time='event_idx', unit='reference_user', condition='type', value='penalty',
                   color='cubehelix', ci=95)
    g.set_xlabel('Event')
    g.set_ylabel('Penalty')
    plt.plot(genuine['event_idx'], genuine['penalty'], label='Genuine')
    plt.axhline(thresh, linestyle='--', color='k', label='Threshold')
    plt.legend(title=None, loc='lower right')
    plt.xticks(np.linspace(0, 500, 6))
    return


def plot_penalty_distribution_example(penalty):
    """
    Show the penalty function for genuine and impostor users
    """
    genuine_idx = penalty['reference_user'] == penalty['query_user']
    genuine = penalty[genuine_idx]
    impostor = penalty[genuine_idx == False]
    thresh = genuine['penalty'].max()
    impostor.loc[:, 'type'] = 'Impostor'
    max_penalty = penalty['penalty'].max()

    # Add the 0 terms at t=0
    genuine = np.concatenate([[0], genuine['penalty'].values])
    impostor = np.concatenate([[0] * len(impostor['reference_user'].unique()), impostor['penalty'].values])

    sns.kdeplot(genuine, color='g',
                shade=True, label='Genuine')
    sns.kdeplot(impostor, color='b',
                shade=True, label='Impostor')
    plt.xlabel('Penalty')
    plt.ylabel('Density')
    plt.axvline(thresh, linestyle='--', color='k', label='Threshold')
    plt.legend(title=None, loc='upper right')
    plt.xlim(0, max_penalty)
    return


def plot_powerlaw_examples(fits, names):
    def plot_fn(ax, i):
        fits[i].plot_ccdf(ax=ax, linewidth=1, color='k')
        fits[i].lognormal.plot_ccdf(ax=ax, linewidth=1, linestyle=':', color='b')
        fits[i].truncated_power_law.plot_ccdf(ax=ax, linewidth=1, linestyle='--', color='r')
        ax.text(0.5, 0.01, names[i], va='bottom', ha='center', transform=ax.transAxes, color='gray', fontsize=15,
                backgroundcolor=ax.get_axis_bgcolor())
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        return

    return plot6(plot_fn, xlabel='$ \\tau $', ylabel='$ \mathbb{P} (\\tau) $', sharey=True)


def plot_powerlaw_exponents(alphas, names, bins):
    def plot_fn(ax, i):
        sns.distplot(alphas[i], ax=ax, bins=bins, norm_hist=True, color=sns.xkcd_rgb['denim blue'])
        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='gray', fontsize=15,
                backgroundcolor=ax.get_axis_bgcolor())
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1.6)
        return

    return plot6(plot_fn, xlabel='$ \\alpha $', ylabel='$ \\mathbb{P} (\\alpha) $', sharex=True, sharey=True)


def plot_allan_factor_examples(Ts, AFs, names, xlabels):
    def plot_fn(ax, i):
        ax.plot(Ts[i], AFs[i], color='k', linewidth=0.1)
        ax.loglog()
        ax.set_xlabel(xlabels[i])
        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='gray', fontsize=15,
                backgroundcolor=ax.get_axis_bgcolor())
        return

    return plot6(plot_fn, xlabel=None, ylabel='AF$ (T) $', sharey=True)


def plot_uniformity_hists(dfs, names, xlabels, max_ticks=10):
    def plot_fn(ax, i):
        if i <= 1:
            sns.distplot(dfs[i].values, ax=ax, bins=np.linspace(0, 1, 11), norm_hist=True,
                         color=sns.xkcd_rgb['denim blue'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
        elif i == 2:
            sns.distplot(dfs[i].values, ax=ax, bins=np.linspace(0, 1, 13), norm_hist=True,
                         color=sns.xkcd_rgb['denim blue'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
        else:
            sns.barplot(dfs[i].index.values, dfs[i].values, color=sns.xkcd_rgb['denim blue'])
            ax.set_xticklabels(['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
            ax.set_ylim(0, 0.3)

        ax.set_xlabel(xlabels[i])
        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='gray', fontsize=15,
                backgroundcolor=ax.get_axis_bgcolor())
        return

    return plot6(plot_fn, xlabel=None, ylabel='Density')


def plot_stationarity_examples(m, names):
    import matplotlib.cm as cm

    def plot_fn(ax, i):
        plt.grid(False)
        plt.imshow(m[i], origin='lower', interpolation='none', cmap=cm.Greys,
                   extent=[0.5, m[i].shape[0] + 0.5, 0.5, m[i].shape[1] + 0.5])

        ax.set_xticks(np.arange(1, m[i].shape[0] + 1))
        ax.set_yticks(np.arange(1, m[i].shape[1] + 1))

        plt.clim(m[i].values.mean() - 4 * m[i].values.std(), m[i].values.mean() + 4 * m[i].values.std())
        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='black', fontsize=15)
        return

    return plot6(plot_fn, xlabel='Train sample', ylabel='Predict sample')


def plot_model_empirical_pdf(df, m, xlim):
    states = m.predict_states_df(df)['state']

    tau_0 = df.loc[states == 0, 'tau'].values
    tau_1 = df.loc[states == 1, 'tau'].values

    upper = xlim
    x = np.linspace(0, upper, 1000)
    fun_0 = m.pdf_fn('tau', hstate=0)
    fun_1 = m.pdf_fn('tau', hstate=1)

    plt.figure(figsize=(8, 6))

    plt.plot(x, fun_0(x), label='Active state', color=sns.xkcd_rgb['denim blue'])
    plt.hist(tau_0, bins=np.linspace(0, upper, 21), normed=True, color=sns.xkcd_rgb['denim blue'], alpha=0.3)

    plt.plot(x, fun_1(x), label='Passive state', color=sns.xkcd_rgb['light red'])
    plt.hist(tau_1, bins=np.linspace(0, upper, 21), normed=True, color=sns.xkcd_rgb['light red'], alpha=0.3)

    plt.xticks(np.linspace(0, upper, 5))
    plt.xlim(0, upper)
    plt.xlabel('$ \\tau $  (ms)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    return


def plot_marginal_pdf_examples(dfs, names, xlims):
    def plot_fn(ax, i):
        m, df = dfs[i]

        _, states = m.predict_states_df(df)

        tau_0 = df.loc[states == 0, 'tau'].values
        tau_1 = df.loc[states == 1, 'tau'].values

        upper = xlims[i]
        x = np.linspace(0, upper, 1000)
        fun_0 = m.pdf_fn('tau', hstate=0)
        fun_1 = m.pdf_fn('tau', hstate=1)

        ax.plot(x, fun_0(x), label='State 0', color=sns.xkcd_rgb['denim blue'])
        ax.hist(tau_0, bins=np.linspace(0, upper, 21), normed=True, color=sns.xkcd_rgb['denim blue'], alpha=0.3)

        ax.plot(x, fun_1(x), label='State 1', color=sns.xkcd_rgb['light red'])
        ax.hist(tau_1, bins=np.linspace(0, upper, 21), normed=True, color=sns.xkcd_rgb['light red'], alpha=0.3)

        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='gray', fontsize=15)

        ax.set_xticks(np.linspace(0, upper, 5))
        ax.set_xlim(0, upper)
        return

    return plot6(plot_fn, xlabel='$ \\tau $', ylabel='Density')


def plot6(plot_fn, xlabel=None, ylabel=None, figsize=(12, 6), sharex=None, sharey=None):
    """
    Generic function to make a 3x2 plot
    """
    fig = plt.figure(figsize=figsize)

    for i in range(6):
        if i == 0:
            ax = fig.add_subplot(2, 3, i + 1)
            if sharex:
                sharex = ax
            if sharey:
                sharey = ax
        else:
            ax = fig.add_subplot(2, 3, i + 1, sharex=sharex, sharey=sharey)

        plot_fn(ax, i)

        if xlabel is not None and (not sharex or i >= 3):
            ax.set_xlabel(xlabel)

        if ylabel is not None and (not sharey or i % 3 == 0):
            ax.set_ylabel(ylabel)

    plt.tight_layout()
    return


def gen_roc():
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresh = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    far = fpr[0]
    frr = 1 - tpr[0]
    roc = pd.DataFrame({'threshold': thresh[0], 'far': far, 'frr': frr})
    roc['threshold'] = (roc['threshold'] - roc['threshold'].min()) / (roc['threshold'].max() - roc['threshold'].min())
    return roc


def plot_continuous_identification_example(n_users=3, n_events=10, seed=2015):
    import matplotlib.cm as cm
    from matplotlib import gridspec

    np.random.seed(seed)
    T = n_events * 10
    time = np.random.randint(0, n_events * 10, n_events)
    user = np.random.randint(1, n_users + 1, n_events)
    zero = np.zeros(n_events)

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[n_users, 1])
    ax0 = plt.subplot(gs[0])

    colors = cm.rainbow(np.linspace(0, 1, n_users))
    for i in range(1, n_users + 1):
        ax0.scatter(time[user == i], user[user == i], color=colors[i - 1])
        ax0.axhline(i, color='k', alpha=0.1)

    ax0.xaxis.set_ticklabels([])
    ax0.set_yticks(np.arange(1, n_users + 1))
    ax0.set_ylabel('User')

    ax1 = plt.subplot(gs[1])
    ax1.axhline(0, color='k', alpha=0.1)
    for i in range(1, n_users + 1):
        ax1.scatter(time[user == i], zero[user == i], color=colors[i - 1])
    ax1.set_yticks([])
    ax1.set_ylabel('Global')
    plt.xlabel('Time')

    plt.tight_layout()
    return


def plot_cdf(df, fit_fn, col):
    hmm = fit_fn(df)
    model_cdf_fn = hmm.cdf_fn(feature=col)
    empir_cdf_fn = ECDF(df[col].values)

    x = np.sort(np.unique(df[col].values))
    x = x[:int(0.9 * len(x))]

    plt.figure(figsize=(8, 8))

    plt.plot(x, empir_cdf_fn(x), color=sns.xkcd_rgb['denim blue'], label='Empirical')
    plt.plot(x, model_cdf_fn(x), 'k--', label='Predicted')
    plt.axis([x.min(), x.max(), 0, 1])
    plt.xlabel('$ \\tau $')
    plt.ylabel('Cumulative distribution')
    plt.legend(loc='lower right')
    return


def plot_pdf(df, fit_fn, col):
    hmm = fit_fn(df)
    model_pdf_fn = hmm.pdf_fn(feature=col)

    x = np.sort(np.unique(df[col].values))
    x = x[:int(0.9 * len(x))]

    plt.figure(figsize=(8, 8))

    sns.distplot(df[col].values, bins=20, norm_hist=True, color=sns.xkcd_rgb['denim blue'], label='Empirical')
    plt.plot(x, model_pdf_fn(x), 'k--', label='Predicted')
    plt.xlim(x.min(), x.max())
    plt.xlabel('$ \\tau $')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    return


def plot_pvalues(pvalues, threshold=0.05):
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(8, 8))

    ecdf = ECDF(pvalues.values.squeeze())
    plt.plot(x, ecdf(x), color=sns.xkcd_rgb['denim blue'])
    plt.plot(x, x, 'k--')
    plt.axvspan(0, threshold, color='k', alpha=0.25)
    plt.minorticks_on()
    plt.xlabel('p-value')
    plt.ylabel('Cumulative distribution')
    return

def plot_hmm_vs_pohmm_pvalues(hmm_pvalues, pohmm_pvalues, threshold=0.05):
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(8, 8))

    plt.plot([], [])
    plt.plot([], [])
    plt.plot([], [])
    plt.plot(x, ECDF(hmm_pvalues.values.squeeze())(x), label='HMM (%.2f)' % ((hmm_pvalues['pvalue'] <= threshold).sum()/len(hmm_pvalues)))
    plt.plot(x, ECDF(pohmm_pvalues.values.squeeze())(x), label='POHMM (%.2f)' % ((pohmm_pvalues['pvalue'] <= threshold).sum()/len(pohmm_pvalues)))
    # plt.plot(x, ECDF(hmm_pvalues.values.squeeze())(x), color=HMM_COLOR, linestyle='-.', label='HMM', linewidth='2.0')
    # plt.plot(x, ECDF(pohmm_pvalues.values.squeeze())(x), color=POHMM_COLOR, label='POHMM')
    plt.plot(x, x, 'k--', linewidth=0.5)
    plt.axvspan(0, threshold, color='k', alpha=0.25)
    plt.minorticks_on()
    plt.xlabel('p-value')
    plt.ylabel('Cumulative distribution')
    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')
    return


def plot_SMAPE_examples(dfs, names, legend_loc):
    def plot_fn(ax, i):
        ax.plot(dfs[i]['event_idx'], pd.expanding_mean(dfs[i]['SMAPE_tau']), label='POHMM', color='k', linewidth=1)
        ax.plot(dfs[i]['event_idx'], pd.expanding_mean(dfs[i]['SMAPE_baseline_tau']), linestyle='--', label='Baseline',
                color='k', linewidth=1)
        ax.text(0.5, 0.95, names[i], va='top', ha='center', transform=ax.transAxes, color='gray', fontsize=15)
        ax.set_ylim(0, 1)
        if i == 5:
            ax.legend(loc=legend_loc)
        return

    return plot6(plot_fn, xlabel='Events', ylabel='SMAPE')


def plot_SMAPE(pred):
    plt.plot(pred['event_idx'], pd.expanding_mean(pred['SMAPE_tau']), label='POHMM')
    plt.plot(pred['event_idx'], pd.expanding_mean(pred['SMAPE_baseline_tau']), linestyle='--', label='Baseline')
    plt.ylim(0, 1)
    plt.legend()
    return


def plot_hist(x, max_x, xlabel, ylabel):
    sns.set(style='darkgrid')
    sns.set_context('notebook', font_scale=3.0)
    fig = plt.figure(figsize=(8, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(x, bins=np.linspace(0.5, max_x + 0.5, max_x + 1), normed=True, color='0.25')
    plt.xticks(range(1, max_x + 1))
    plt.xlim([0, max_x + 1])
    return


def run(plot_fun):
    globals()[plot_fun]()
    save_fig(plot_fun)
    return
