import numpy as np
import pandas as pd
from pohmm import Pohmm
from scipy import interp
from itertools import chain
from scipy.stats import wilcoxon
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.metrics import auc, accuracy_score

from .io import load_data, load_results, save_results, ProgressBar
from .data import preprocess_data, MOBILE_SENSORS, DATASETS
from .plotting import *


def leave_one_out(samples_per_user):
    folds = []
    for i in range(samples_per_user):
        folds.append((np.r_[np.arange(i), np.arange(i + 1, samples_per_user)],
                      np.r_[i],
                      np.r_[i]))

    return folds


VALIDATION = {
    'password': [(np.arange(150, 200), np.arange(200, 400), np.arange(200, 400))],
    'keypad': leave_one_out(20),
    'fixed_text': leave_one_out(4),
    'free_text': leave_one_out(6),
    'mobile': leave_one_out(20)
}


def pohmm_factory(df):
    emissions = []
    for col in df.columns.difference(['event']):
        if col in ['tau', 'duration']:
            emissions.append((col, 'lognormal'))
        else:
            emissions.append((col, 'normal'))

    hmm = Pohmm(n_hidden_states=2, init_spread=2, thresh=1e-6, max_iter=1000,
                emissions=emissions, smoothing='freq')
    hmm.fit_df(list(zip(*df.groupby(level=[0, 1])))[1])
    return hmm


def stratified_kfold(df, nfolds):
    """
    Create stratified k-folds
    """
    sessions = pd.DataFrame.from_records(list(df.index.unique())).groupby(0).apply(lambda x: x[1].unique())
    sessions.apply(lambda x: np.random.shuffle(x))
    folds = []
    for i in range(nfolds):
        idx = sessions.apply(lambda x: pd.Series(x[i * (len(x) / nfolds):(i + 1) * (len(x) / nfolds)]))
        idx = pd.DataFrame(idx.stack().reset_index(level=1, drop=True)).set_index(0, append=True).index.values
        folds.append(df.loc[idx])
    return folds


def cv_session_scores(folds, model_factory):
    """
    Obtain identification and verification results using stratified k-fold cross validation and a model that scores a sample

    fit_model_fn should be a function that takes all the samples from a single user and returns a fitted model
    score_model_fn should be a function that takes a model and a single sample and scores the sample for the model
    """
    results = []
    n_folds = len(folds)
    for i in range(n_folds):
        print('\nFold %d of %d' % (i + 1, n_folds))

        reference, genuine, impostor = folds[i]
        reference_users = reference.index.get_level_values(0).unique()

        work_done = 0
        work = len(reference_users) + len(genuine.index.unique()) + len(impostor.index.unique())
        progress = ProgressBar(work)

        models = {}
        for reference_user, reference_data in reference.groupby(level=[0]):
            models[reference_user] = model_factory(reference_data)
            work_done += 1
            progress.animate(work_done)

        for (reference_user, query_user, query_session), query_data in chain(genuine.groupby(level=[0, 1, 2]),
                                                                             impostor.groupby(level=[0, 1, 2])):
            results.append((i, reference_user, query_user, query_session,
                            models[reference_user].score_df(query_data)))
            work_done += 1
            progress.animate(work_done)
    print()
    scores = pd.DataFrame(results, columns=['fold', 'reference_user', 'query_user', 'query_session', 'score'])
    #     scores.set_index(['fold','reference_user','query_user','query_session'], inplace=True)
    return scores


def model_scores(df, model):
    if df.index.nlevels > 1:
        level = np.arange(df.index.nlevels).tolist()
    else:
        level = 0

    def loglik(x):
        m = model(x)
        return m.logprob_

    scores = df.groupby(level=level).apply(loglik)
    scores = pd.DataFrame(scores)
    scores.columns = ['loglik']
    return scores


def cv_event_scores(folds, model, show_progress=True):
    """
    Obtain identification and verification results using stratified k-fold cross validation and a model that scores a sample

    Creates a dataframe with cols: fold, reference_user, query_user, query_session, event_idx
    Args:
        folds: list of folds
        model: function that takes all the samples from a single user and returns a fitted model
    """
    scores = []
    n_folds = len(folds)
    for i in range(n_folds):
        if show_progress:
            print('\nFold %d of %d' % (i + 1, n_folds))

        reference, genuine, impostor = folds[i]
        reference_users = reference.index.get_level_values(0).unique()

        work_done = 0
        work = len(reference_users) + len(genuine.index.unique()) + len(impostor.index.unique())
        progress = ProgressBar(work)
        if show_progress:
            progress.animate(work_done)

        models = {}
        for reference_user, reference_data in reference.groupby(level=[0]):
            models[reference_user] = model(reference_data)
            work_done += 1
            if show_progress:
                progress.animate(work_done)

        for (reference_user, query_user, query_session), query_data in chain(genuine.groupby(level=[0, 1, 2]),
                                                                             impostor.groupby(level=[0, 1, 2])):
            score = models[reference_user].score_events_df(query_data.reset_index(drop=True))
            state = models[reference_user].predict_states_df(query_data.reset_index(drop=True))

            df = pd.DataFrame({'fold': i,
                               'reference_user': reference_user,
                               'query_user': query_user,
                               'query_session': query_session,
                               'event_idx': np.arange(len(query_data)),
                               'event': query_data['event'].values,
                               'score': score['score'],
                               'state': state['state'],
                               },
                              columns=['fold', 'reference_user', 'query_user', 'query_session', 'event_idx',
                                       'event', 'score', 'state'])
            scores.append(df)
            work_done += 1
            if show_progress:
                progress.animate(work_done)

    scores = pd.concat(scores).reset_index(drop=True)
    scores['rank'] = scores.groupby(['fold', 'query_user',
                                     'query_session', 'event_idx'])['score'].rank(ascending=False) - 1
    return scores


def normalize_session_scores(session_scores, pivot=['fold', 'query_user', 'query_session'], method='minmax', h=2):
    def _norm(df):
        if method is None:
            df['nscore'] = df['score']
            return df

        if method == 'minmax':
            lower = df['score'].min()
            upper = df['score'].max()
        elif method == 'stddev':
            lower = df['score'].mean() - h * df['score'].std()
            upper = df['score'].mean() + h * df['score'].std()

        df['nscore'] = np.minimum(np.maximum((df['score'] - lower) / (upper - lower), 0), 1)
        return df

    session_scores = session_scores.groupby(pivot).apply(_norm)
    return session_scores


def session_identification(session_scores):
    """

    """
    ide = session_scores.groupby(['fold', 'query_user', 'query_session']).apply(
        lambda x: x.iloc[np.argmax(x['score'].values)][['reference_user']])
    ide.columns = ['result']
    ide = ide.reset_index()
    return ide


def roc_curve(y_true, y_score):
    """
    See sklearn.metrics.roc_curve
    """
    from sklearn.metrics import roc_curve as _roc_curve
    fpr, tpr, thresholds = _roc_curve(y_true, y_score, drop_intermediate=True)
    return fpr, 1 - tpr, thresholds


def session_roc(session_scores, pivot='fold'):
    """

    """
    # Generate an ROC curve for each fold, ordered by increasing threshold
    roc = session_scores.groupby(pivot).apply(
        lambda x: pd.DataFrame(np.c_[roc_curve((x['query_user'] == x['reference_user']).values.astype(np.int32),
                                               x['nscore'].values.astype(np.float32))][::-1],
                               columns=['far', 'frr', 'threshold']))

    # interpolate to get the same threshold values in each fold
    thresholds = np.sort(roc['threshold'].unique())
    roc = roc.groupby(level=pivot).apply(lambda x: pd.DataFrame(np.c_[thresholds,
                                                                      interp(thresholds, x['threshold'], x['far']),
                                                                      interp(thresholds, x['threshold'], x['frr'])],
                                                                columns=['threshold', 'far', 'frr']))
    roc = roc.reset_index(level=1, drop=True).reset_index()
    return roc


def continuous_identification(scores):
    """

    """
    ide = scores.groupby(['fold', 'query_user', 'query_session', 'event_idx']).apply(
        lambda x: x.iloc[np.argmax(x['score'].values)][['reference_user']])
    ide.columns = ['result']
    ide = ide.reset_index()
    return ide


def scores_penalty(scores, penalty_fun='sum', window=25):
    """

    """

    def _penalty(df):
        if penalty_fun == 'sum':
            p = df['rank'].rolling(window=window, center=False).sum()
            p[:window] = df['rank'].values[:window].cumsum()
        elif penalty_fun == 'sumexp':
            p = (np.exp(df['rank']) - 1).rolling(window=window, center=False).sum()
            p[:window] = (np.exp(df['rank']) - 1)[:window].cumsum()

        df['penalty'] = p
        return df

    penalty = scores.copy().groupby(['fold', 'reference_user', 'query_user', 'query_session']).apply(_penalty)
    return penalty


def continuous_verification(penalty):
    """
    Determine the maximum lockout time for each impostor/query sample
    """

    genuine_idx = penalty['reference_user'] == penalty['query_user']
    genuine = penalty[genuine_idx]
    lockout = genuine.groupby(['query_user', 'query_session']).max()[['penalty']]
    lockout = pd.DataFrame(lockout)
    lockout.columns = ['threshold']

    impostor = penalty[~genuine_idx]

    def _mrt(df):
        # thresh = lockout.loc[tuple(df.iloc[0][['query_user', 'query_session']].values)].squeeze()
        thresh = 645
        reject = (df['penalty'] > thresh)
        return np.where(reject)[0].min() if reject.any() else len(reject)

    mrt = impostor.groupby(['reference_user', 'query_user', 'query_session']).apply(_mrt).reset_index()
    mrt.columns = ['reference_user', 'query_user', 'query_session', 'mrt']

    amrt = mrt.groupby(['query_user', 'query_session'])['mrt'].mean()
    amrt.columns = ['amrt']

    results = pd.concat([amrt, lockout], axis=1).reset_index()
    return results


def continuous_verification(penalty):
    """
    Determine the maximum lockout time for each impostor/query sample
    """

    genuine_idx = penalty['reference_user'] == penalty['query_user']
    genuine = penalty[genuine_idx]
    lockout = genuine.groupby(['query_user', 'query_session']).max()[['penalty']]
    lockout = pd.DataFrame(lockout)
    lockout.columns = ['threshold']

    impostor = penalty[genuine_idx == False]

    def _mrt(df):
        thresh = lockout.loc[tuple(df.iloc[0][['query_user', 'query_session']].values)].squeeze()
        reject = (df['penalty'] > thresh)
        return np.where(reject)[0].min() if reject.any() else len(reject)

    mrt = impostor.groupby(['reference_user', 'query_user', 'query_session']).apply(_mrt).reset_index()
    mrt.columns = ['reference_user', 'query_user', 'query_session', 'mrt']

    amrt = mrt.groupby(['query_user', 'query_session'])['mrt'].mean()
    amrt.columns = ['amrt']

    results = pd.concat([amrt, lockout], axis=1).reset_index()
    return results


def ACC(ide):
    """
    Obtain rank-n classification accuracy for each fold
    """
    return accuracy_score(ide['query_user'].values, ide['result'].values)


def EER(roc):
    """
    Obtain the EER for one fold
    """
    far, frr = roc['far'].values, roc['frr'].values

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    def seg_intersect(a1, a2, b1, b2):
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom) * db + b1

    d = far <= frr
    idx = np.diff(d).nonzero()[0][0]
    return seg_intersect(np.array([idx, far[idx]]),
                         np.array([idx + 1, far[idx + 1]]),
                         np.array([idx, frr[idx]]),
                         np.array([idx + 1, frr[idx + 1]]))[1]


def AUC(roc):
    """
    Area under the ROC curve
    """
    return auc(roc['frr'].values, roc['far'].values)


def SMAPE(ground_truth, predictions):
    """
    Symmetric mean absolute prediction error
    """
    return np.abs((ground_truth - predictions) / (ground_truth + predictions))


def split_dataset(df, template_reps, genuine_reps, impostor_reps):
    df_template = df[df.index.get_level_values(1).isin(template_reps)]
    df_genuine = df[df.index.get_level_values(1).isin(genuine_reps)]
    df_impostor = df[df.index.get_level_values(1).isin(impostor_reps)]

    df_genuine.index.names = ['reference_user', 'session']
    df_genuine = df_genuine.reset_index()
    df_genuine['query_user'] = df_genuine['reference_user']
    df_genuine = df_genuine.set_index(['reference_user', 'query_user', 'session'])

    df_impostor.index.names = ['reference_user', 'session']
    df_impostor = df_impostor.reset_index()
    df_impostor['query_user'] = df_impostor['reference_user']
    df_impostor = df_impostor.set_index(['reference_user', 'query_user', 'session'])

    dfs_impostor = []
    for user in df.index.get_level_values(0).unique():
        df_tmp = df_impostor.drop(user, level=0).reset_index().copy()
        df_tmp['reference_user'] = user
        dfs_impostor.append(df_tmp)

    df_impostor = pd.concat(dfs_impostor).set_index(['reference_user', 'query_user', 'session'])

    return df_template, df_genuine, df_impostor


def dataset_classification_results(dataset, event, features=['tau', 'duration'],
                                   model_factory_fn=pohmm_factory, out_name=None):
    """
    Obtain results for a given dataset and features conditioned on the event column.
    """
    print('Running:', out_name, flush=True)

    # Load and preprocess the dataset
    df = load_data(dataset)
    df = preprocess_data(df, event, features)

    # Create the validation folds
    folds = [split_dataset(df, *sessions) for sessions in VALIDATION[dataset]]
    scores = cv_event_scores(folds, model_factory_fn)
    save_results(scores, out_name + '_event_scores')

    # Aggregate and normalize the event scores within each session
    session_scores = scores.groupby(['fold', 'reference_user',
                                     'query_user', 'query_session'])['score'].sum().reset_index()
    session_scores = normalize_session_scores(session_scores)
    save_results(session_scores, out_name + '_session_scores')

    # Session and continuous identification, verification results
    session_ide = session_identification(session_scores)
    session_ver = session_roc(session_scores)

    continuous_ide = continuous_identification(scores)  # Identification of each event
    penalty = scores_penalty(scores)
    continuous_ver = continuous_verification(penalty)  # Minimum rejection time

    # Summarize of session results
    session_acc = session_ide.groupby('fold').apply(ACC).describe()
    session_eer = session_ver.groupby('fold').apply(EER).describe()
    session_auc = session_ver.groupby('fold').apply(AUC).describe()

    # User-dependent EER is obtained by deriving an ROC curve for each user
    user_eer = session_roc(session_scores, pivot='reference_user').groupby('reference_user').apply(EER).describe()
    user_acc = session_ide.groupby('query_user').apply(ACC).describe()

    # Summarize continuous results, CI by session
    continuous_acc = continuous_ide.groupby(['query_user', 'query_session']).apply(ACC).describe()

    # Maximum lockout time, averaged for each session (against all reference users), CI by session
    continuous_amrt = continuous_ver['amrt'].describe()

    summary = pd.concat([session_acc, user_acc, session_eer, user_eer, session_auc, continuous_acc, continuous_amrt],
                        axis=1)
    summary.columns = ['ACC', 'U-ACC', 'EER', 'U-EER', 'AUC', 'CIA', 'AMRT']
    save_results(summary, out_name + '_summary')
    print(summary)

    event_scores = load_results(out_name + '_event_scores')
    penalty = scores_penalty(event_scores)

    # Plot a penalty function example
    penalty = penalty.set_index(['query_user', 'query_session'])
    penalty_example = penalty.loc[np.random.choice(penalty.index.unique())].reset_index()
    plot_penalty_example(penalty_example)
    save_fig(out_name + '_penalty_example')

    plot_penalty_distribution_example(penalty_example)
    save_fig(out_name + '_penalty_distribution_example')

    # plot the error and ROC curves
    plot_error(session_ver)
    save_fig(out_name + '_error')

    plot_roc(session_ver)
    save_fig(out_name + '_roc')
    return


def dataset_prediction_results(dataset, event, model_factory_fn=pohmm_factory,
                               min_history=90, max_history=None, out_name=None):
    """
    Obtain predictions for each model.

    Create stratified folds
    Train on 1-n_folds. Use the last fold to make predictions for each event
    """
    print('Running:', out_name, flush=True)

    # Load and preprocess the dataset
    df = load_data(dataset)
    # from .data import reduce_dataset
    # df = reduce_dataset(df, num_users=5, min_samples=1, max_samples=1)
    df = preprocess_data(df, event, ['tau'])

    # fold, ref user, query user, query session, into future, event, ground truth, prediction
    baseline_col = 'baseline_tau'
    prediction_col = 'prediction_tau'

    work_done = 0
    work = len(df.index.unique())
    progress = ProgressBar(work)
    progress.animate(work_done)

    def _predictions(df):
        if max_history is None:
            upper = len(df) - 1
        else:
            upper = min(max_history, len(df) - 1)

        results = []
        for i in range(min_history, upper + 1):
            hmm = model_factory_fn(df[:i])
            pred = hmm.predict_df(df[:i], next_pstate=df.iloc[i]['event'])[0]
            # pred = hmm.predict_df(df[:i])[0]
            baseline_pred = df['tau'].values[:i].mean(axis=0)
            results.append([i, df.iloc[i]['event'], df.iloc[i]['tau'], pred, baseline_pred])

        nonlocal work_done
        work_done += 1
        progress.animate(work_done)

        results = pd.DataFrame(results, columns=['event_idx', 'event', 'tau', prediction_col, baseline_col])
        return results

    pred = df.groupby(level=[0, 1]).apply(_predictions)
    pred['SMAPE_tau'] = SMAPE(pred['tau'], pred[prediction_col])
    pred['SMAPE_baseline_tau'] = SMAPE(pred['tau'], pred[baseline_col])
    pred = pred.reset_index(level=df.index.nlevels, drop=True)

    save_results(pred, out_name + '_predictions')
    return


def manhattan_factory(df):
    class Classifier(object):
        def fit_df(self, df):
            self.template = df.mean(axis=0)

        def score_df(self, df):
            return - (self.template - df).abs().sum(axis=1).values.squeeze()

    clf = Classifier()
    clf.fit_df(df)
    return clf


def svm_factory(df):
    class Classifier(object):
        def fit_df(self, df):
            self.model = OneClassSVM()
            self.model.fit(df.values)

        def score_df(self, df):
            return self.model.decision_function(df.values).squeeze()

    clf = Classifier()
    clf.fit_df(df)
    return clf


def gmm_factory(df):
    class Classifier(object):
        def fit_df(self, df):
            df = df[df.columns.difference(['event'])]
            n_components = int(round(np.sqrt(df.groupby(level=[0, 1]).size().mean())))
            self.model = GMM(n_components=n_components, covariance_type='spherical', min_covar=0.01)
            self.model.fit(df.values)

        def score_events_df(self, df):
            df = df[df.columns.difference(['event'])]
            df['score'] = self.model.score(df.values)
            return df

        def predict_states_df(self, df):
            df['state'] = 0
            return df

    clf = Classifier()
    clf.fit_df(df)
    return clf


def feature_vector_results(dataset, features, model_factory, out_name):
    print('Running:', out_name, flush=True)

    df = load_data(features)

    folds = [split_dataset(df, *sessions) for sessions in VALIDATION[dataset]]
    scores = cv_session_scores(folds, model_factory)

    session_scores = normalize_session_scores(scores)
    save_results(session_scores, out_name + '_session_scores')

    # Session and continuous identification, verification results
    session_ide = session_identification(session_scores)
    session_ver = session_roc(session_scores)

    # Summarize of session results
    session_acc = session_ide.groupby('fold').apply(ACC).describe()
    session_eer = session_ver.groupby('fold').apply(EER).describe()
    session_auc = session_ver.groupby('fold').apply(AUC).describe()

    # User-dependent EER is obtained by deriving an ROC curve for each user
    user_eer = session_roc(session_scores, pivot='reference_user').groupby('reference_user').apply(EER).describe()
    user_acc = session_ide.groupby('query_user').apply(ACC).describe()

    summary = pd.concat([session_acc, user_acc, session_eer, user_eer, session_auc], axis=1)
    summary.columns = ['ACC', 'U-ACC', 'EER', 'U-EER', 'AUC']
    save_results(summary, out_name + '_summary')
    print(summary)


def classification_results(seed=1234):
    np.random.seed(seed)

    for dataset in DATASETS:
        dataset_classification_results(dataset, 'keyname', out_name='%s_pohmm' % dataset)
        dataset_classification_results(dataset, 'none', out_name='%s_hmm' % dataset)

    dataset_classification_results('mobile', 'keyname',
                                   features=['tau', 'duration'] + MOBILE_SENSORS,
                                   out_name='mobile_sensor_pohmm')
    dataset_classification_results('mobile', 'none',
                                   features=['tau', 'duration'] + MOBILE_SENSORS,
                                   out_name='mobile_sensor_hmm')

    for dataset in ['fixed_text', 'free_text']: #DATASETS:
        # feature_vector_results(dataset, '%s_features' % dataset, manhattan_factory, out_name='%s_manhattan' % dataset)
        feature_vector_results(dataset, '%s_scaled_features' % dataset, manhattan_factory,
                               out_name='%s_scaled_manhattan' % dataset)
        feature_vector_results(dataset, '%s_normed_features' % dataset, svm_factory, out_name='%s_svm' % dataset)

    feature_vector_results('mobile', 'mobile_sensor_features', manhattan_factory,
                           out_name='mobile_sensor_manhattan')
    feature_vector_results('mobile', 'mobile_sensor_scaled_features', manhattan_factory,
                           out_name='mobile_sensor_scaled_manhattan')
    feature_vector_results('mobile', 'mobile_sensor_normed_features', svm_factory,
                           out_name='mobile_sensor_svm')


def prediction_results(seed=1234):
    np.random.seed(seed)

    dataset_prediction_results('fixed_text', 'keyname', out_name='fixed_text_pohmm', min_history=50, max_history=None)
    dataset_prediction_results('fixed_text', 'none', out_name='fixed_text_hmm', min_history=50, max_history=None)

    np.random.seed(seed)
    dataset_prediction_results('free_text', 'keyname', out_name='free_text_pohmm', min_history=450, max_history=None)
    dataset_prediction_results('free_text', 'none', out_name='free_text_hmm', min_history=450, max_history=None)


def plot_pohmm_example(dataset, seed=1234):
    np.random.seed(seed)
    df = load_data(dataset)
    df = df[df.index.get_level_values(0) == np.random.choice(df.index.get_level_values(0).unique())]
    df = preprocess_data(df, 'keyname', ['tau'])
    m = pohmm_factory(df)
    plot_model_empirical_pdf(df, m, 1000)
    save_fig('%s_pohmm_example' % dataset)


def plot_montecarlo_hmm_vs_pohmm(dataset):
    hmm_pvalues = load_results('%s_hmm_montecarlo_pvalues' % dataset)
    pohmm_pvalues = load_results('%s_pohmm_montecarlo_pvalues' % dataset)
    plot_hmm_vs_pohmm_pvalues(hmm_pvalues, pohmm_pvalues)
    save_fig('%s_hmm_vs_pohmm_pvalues' % dataset)


def plot_roc_curves_hmm_vs_pohmm(dataset):
    if dataset == 'password':
        pivot = 'reference_user'
    else:
        pivot = 'fold'

    manhattan_roc = session_roc(load_results('%s_manhattan_session_scores' % dataset), pivot)
    scaled_manhattan_roc = session_roc(load_results('%s_scaled_manhattan_session_scores' % dataset), pivot)
    one_class_svm = session_roc(load_results('%s_svm_session_scores' % dataset), pivot)
    hmm_roc = session_roc(load_results('%s_hmm_session_scores' % dataset), pivot)
    pohmm_roc = session_roc(load_results('%s_pohmm_session_scores' % dataset), pivot)

    plot_roc([('Manhattan', manhattan_roc),
              ('Manhattan (scaled)', scaled_manhattan_roc),
              ('SVM (one-class)', one_class_svm),
              ('HMM', hmm_roc),
              ('POHMM', pohmm_roc)], 'Model', pivot)

    save_fig(dataset + '_roc')


def summary_table(m, threshold=0.05):
    rows = []

    if m == 'AMRT':
        SYSTEMS = ['hmm', 'pohmm']
        COLUMNS = ['dataset', 'HMM', 'POHMM']
    else:
        SYSTEMS = ['manhattan', 'scaled_manhattan', 'svm', 'hmm', 'pohmm']
        COLUMNS = ['dataset', 'Manhattan', 'Manhattan (scaled)', 'SVM (one-class)', 'HMM', 'POHMM']

    for dataset in ['password', 'keypad', 'mobile', 'mobile_sensor', 'fixed_text', 'free_text']:
        row = []

        if ((m == 'EER') or (m == 'ACC')) and (dataset == 'password'):
            measure = 'U-' + m
        else:
            measure = m

        means = []
        system_measures = []
        for system in SYSTEMS:
            session_scores = load_results('%s_%s_session_scores' % (dataset, system))

            if measure == 'U-ACC':
                measures = session_identification(session_scores).groupby('query_user').apply(ACC)
            elif measure == 'U-EER':
                measures = session_roc(session_scores, pivot='reference_user').groupby('reference_user').apply(EER)
            elif measure == 'ACC':
                measures = session_identification(session_scores).groupby('fold').apply(ACC)
            elif measure == 'EER':
                measures = session_roc(session_scores, pivot='fold').groupby('fold').apply(EER)
            elif measure == 'AMRT':
                scores = load_results('%s_%s_event_scores' % (dataset, system))
                penalty = scores_penalty(scores)
                continuous_ver = continuous_verification(penalty)
                measures = continuous_ver['amrt']

            system_measures.append(measures.values)
            means.append(measures.mean())
            row.append('%.3f (%.3f)' % (measures.mean(), measures.std()))

        means = np.array(means)

        if 'ACC' in measure:
            idx = np.argmax(means)
        else:
            idx = np.argmin(means)

        row[idx] = '*' + row[idx] + '*'

        for i in range(len(system_measures)):
            if i == idx:
                continue
            _, pvalue = wilcoxon(system_measures[idx], system_measures[i])
            if pvalue > threshold/(len(system_measures) - 1):
                row[i] = '*' + row[i] + '*'

        rows.append([dataset] + row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.set_index('dataset')
    save_results(df, 'summary_%s' % m)
