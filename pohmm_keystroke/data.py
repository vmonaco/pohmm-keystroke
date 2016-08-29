import os
import numpy as np
import pandas as pd
from collections import defaultdict

from .io import save_data, load_data, exists_data, save_results
from . import RAW_DATA_DIR

DATASETS = ['password', 'keypad', 'fixed_text', 'free_text', 'mobile']

MOBILE_SENSORS = ['pressure', 'tool_major', 'x', 'x_acceleration', 'x_rotation', 'y', 'y_acceleration',
                  'y_rotation', 'z_acceleration', 'z_rotation']

SUMMARY_COLS = ['Users', 'Samples/user', 'Total events', 'Min events/user', 'Max events/user',
                'Events/sample', 'Mean user freq', 'Mean user period', 'Global freq', 'Mean global period']

KEYGROUP = {
    # 'backspace': 'backspace',
    'space': 'space',
    'shift': 'shift',
    'period': 'period',
    'comma': 'comma'
}

for k in list('qwertasdfgzxcvb'):
    KEYGROUP[k] = 'left_letter'

for k in list('yuiophjklnm'):
    KEYGROUP[k] = 'right_letter'

FEATURE_FUNS = defaultdict(lambda: lambda df, col: df[col])
FEATURE_FUNS.update({
    # Timing features
    'tau': lambda df, col: df['timepress'].diff().fillna(np.median(df['timepress'].diff().dropna())),
    'duration': lambda df, col: df['timerelease'] - df['timepress'],

    # Event type functions
    'none': lambda df, col: np.ones(len(df)),
    'keygroup': lambda df, col: df['keyname'].map(KEYGROUP).fillna('other'),
    'position': lambda df, col: np.arange(len(df)),
})


def preprocess_data(df, event_col, feature_cols):
    def pp_fun(x, feature_cols=feature_cols):
        x['event'] = FEATURE_FUNS[event_col](x, event_col)

        for col in feature_cols:
            x[col] = FEATURE_FUNS[col](x, col)

        return x[['event'] + feature_cols]

    if df.index.nlevels > 1:
        level = np.arange(df.index.nlevels).tolist()
    else:
        level = 0

    df = df.groupby(level=level).apply(pp_fun)
    return df


def reduce_dataset(df, num_users=None,
                   min_samples=None, max_samples=None,
                   min_obs=None, max_obs=None):
    '''
    Reducing the size of a dateset is a common operation when a certain number
    of observations, samples, or users is desired. This function limits each
    of these by attempting to satisfy the constraints in the following order:
    
        num observations
        num samples
        num users

    '''

    if max_obs:
        df = df.groupby(level=[0, 1]).apply(lambda x: x[:max_obs]).reset_index(level=[2, 3], drop=True)

    num_obs = df.groupby(level=[0, 1]).size()

    if min_obs:
        num_obs = num_obs[num_obs >= min_obs]

    num_samples = num_obs.groupby(level=0).size()

    if min_samples:
        num_samples = num_samples[num_samples >= min_samples]

    if num_users and num_users < len(num_samples):
        users = np.random.permutation(num_samples.index.values)[:num_users]
    else:
        users = num_samples.index.values

    num_obs = num_obs.loc[users.tolist()]

    if max_samples:
        num_obs = num_obs.groupby(level=0).apply(
            lambda x: x.loc[np.random.permutation(np.sort(x.index.unique()))[:max_samples]]).reset_index(level=1,
                                                                                                         drop=True)

    df = df.loc[num_obs.index].sort_index()

    return df


def _filter(df, max_dups=4, max_pause=6e4):
    # Drop sessions with many duplicate times
    s = df.reset_index().groupby(['user', 'session', 'timepress']).size()
    s = s[s > max_dups].reset_index(level=2, drop=True)
    dropme = s.index.unique()
    df = df.drop(dropme)

    # Drop sessions with more than 5 minute pauses
    s = df.groupby(level=[0, 1]).apply(lambda x: np.any(x['timepress'].diff() > max_pause))
    s = s[s]
    dropme = s.index.unique()
    df = df.drop(dropme)

    # Drop 0 durations
    df = df[df['timerelease'] - df['timepress'] > 0]

    # Separate duplicate key presses by at least 1 ms
    while np.any(df.groupby(level=[0, 1]).apply(lambda x: x['timepress'].diff() == 0)):
        def _inc_timepress_dups(x):
            idx = x['timepress'].diff().fillna(1) == 0
            x.loc[idx, 'timepress'] += 1
            x.loc[idx, 'timerelease'] += 1
            return x.reset_index()

        df = df.groupby(level=[0, 1]).apply(_inc_timepress_dups).set_index(['user', 'session'])

    df = df.groupby(level=[0, 1]).apply(lambda x: x.sort_values('timepress'))

    return df


def _normalize(df):
    def norm_session_times(x):
        t0 = x.iloc[0]['timepress']
        x['timepress'] -= t0
        x['timerelease'] -= t0
        return x

    df = df.groupby(level=[0, 1]).apply(norm_session_times)

    df = df.reset_index()
    df['user'] = df['user'].map(dict(zip(df['user'].unique(), range(len(df['user'].unique())))))

    def renumber_sessions(x):
        x['session'] = x['session'].map(dict(zip(sorted(x['session'].unique()), range(len(x['session'].unique())))))
        return x

    df = df.groupby('user').apply(renumber_sessions).set_index(['user', 'session'])
    df = df.sort_index()
    return df


def preprocess_password(fname_in):
    def process_row(idx_row):
        idx, row = idx_row

        timepress = 1000 * np.r_[0, row[4::3].astype(float).values].cumsum()
        timerelease = timepress + 1000 * row[3::3].astype(float).values

        keyname = list('.tie5Roanl') + ['enter']

        return pd.DataFrame.from_items([
            ('user', [row['subject']] * 11),
            ('session', [row['sessionIndex'] * 100 + row['rep']] * 11),
            ('keyname', keyname),
            ('timepress', timepress),
            ('timerelease', timerelease)
        ])

    df = pd.concat(map(process_row, pd.read_csv(fname_in).iterrows())).set_index(['user', 'session'])

    df = _normalize(df)
    save_data(df, 'password')
    return


def preprocess_keypad(fname_in):
    df = pd.read_csv(fname_in, index_col=[0, 1])

    # Discard incorrect entries
    keynames = ['numpad_%s' % s for s in '9141937761'] + ['enter']
    df = df.groupby(level=[0, 1]).filter(lambda x: (len(x) == 11) and (x['keyname'] == keynames).all())

    df = _normalize(df)
    save_data(df, 'keypad')
    return


def preprocess_fixed_text(fname1_in, fname2_in, num_samples=4, num_obs=100):
    df1 = pd.read_csv(fname1_in, index_col=[0, 1])
    df2 = pd.read_csv(fname2_in, index_col=[0, 1])

    df = pd.concat([df1[df1['inputtype'] == 'fixed'][['keyname', 'timepress', 'timerelease']],
                    df2[['keyname', 'timepress', 'timerelease']]])

    df = _filter(df)
    df = reduce_dataset(df, min_samples=num_samples, max_samples=num_samples, min_obs=num_obs, max_obs=num_obs)
    df = _normalize(df)
    save_data(df, 'fixed_text')
    return


def preprocess_free_text(fname_in, num_samples=6, num_obs=500):
    df = pd.read_csv(fname_in, index_col=[0, 1])

    df = df[df['inputtype'] == 'free'][['keyname', 'timepress', 'timerelease']]

    df = _filter(df)
    df = reduce_dataset(df, min_samples=num_samples, max_samples=num_samples, min_obs=num_obs, max_obs=num_obs)
    df = _normalize(df)
    save_data(df, 'free_text')
    return


def preprocess_mobile(fname_in, num_samples=20, num_users=None):
    df = pd.read_csv(fname_in, index_col=[0, 1])
    entities = {57: '9', 49: '1', 52: '4', 51: '3', 55: '7', 54: '6', 10: 'enter'}
    df['keyname'] = df['entity'].map(entities)
    df = df.dropna()

    # Only correct entries
    keynames = np.repeat(np.array(['9', '1', '4', '1', '9', '3', '7', '7', '6', '1', 'enter']), 2)
    df = df.groupby(level=[0, 1]).filter(
        lambda x: (len(x) == 22) and (x['keyname'] == keynames).all() and (x[::2]['action'] == 'press').all() and (
            x[1::2]['action'] == 'release').all())

    COLS = ['pressure',
            'tool_major',
            'x',
            'x_acceleration',
            'x_rotation',
            'y',
            'y_acceleration',
            'y_rotation',
            'z_acceleration',
            'z_rotation']

    df = df.reset_index()
    press = df[df['action'] == 'press']
    release = df[df['action'] == 'release']
    press.columns = ['press_{}'.format(c) for c in df.columns]
    release.columns = ['release_{}'.format(c) for c in df.columns]
    release.index = release.index - 1
    df = pd.concat([press, release], axis=1)

    df['user'] = df['press_user']
    df['session'] = df['press_session']
    df['keyname'] = df['press_keyname']
    df['timepress'] = df['press_time']
    df['timerelease'] = df['release_time']

    for c in COLS:
        df[c] = df[['press_{}'.format(c), 'release_{}'.format(c)]].mean(axis=1)

    df.set_index(['user', 'session'], inplace=True)
    df = df[['keyname', 'timepress', 'timerelease'] + COLS]

    df = reduce_dataset(df, num_users=num_users, min_samples=num_samples, max_samples=num_samples)
    df = _normalize(df)
    save_data(df, 'mobile')
    return


def preprocess_raw_data(seed=1234):
    np.random.seed(seed)
    if not exists_data('password'):
        preprocess_password(os.path.join(RAW_DATA_DIR, 'DSL-StrongPasswordData.csv'))

    if not exists_data('keypad'):
        preprocess_keypad(os.path.join(RAW_DATA_DIR, 'keypad.csv'))

    if not exists_data('fixed_text'):
        preprocess_fixed_text(os.path.join(RAW_DATA_DIR, 'villani.csv'), os.path.join(RAW_DATA_DIR, 'nursery.csv'))

    if not exists_data('free_text'):
        preprocess_free_text(os.path.join(RAW_DATA_DIR, 'villani.csv'))

    if not exists_data('mobile'):
        preprocess_mobile(os.path.join(RAW_DATA_DIR, 'phonenumber_sept2014.csv'))


def _dataset_summary(dataset):
    df = load_data(dataset)

    s = df.groupby(level=[0, 1]).size()
    su = df.groupby(level=0).size()
    mean_tau = df.groupby(level=[0, 1]).apply(lambda x: x['timepress'].diff().dropna().mean())
    mean_user_freq = (1 / mean_tau).mean()
    global_freq = len(df) / (df['timepress'].max() - df['timepress'].min())
    mean_global_period = df['timepress'].sort(inplace=False).diff().dropna().mean()

    row = [
        '%d' % len(su),
        '%d' % s.groupby(level=0).size().mean(),
        '%d' % len(df),
        '%d' % su.min(),
        '%d' % su.max(),
        '%.4f +/- %.4f' % (s.mean(), s.std()),
        '%.4f' % mean_user_freq,
        '%.4f' % mean_tau.mean(),
        '%.4f' % global_freq,
        '%.4f' % mean_global_period
    ]
    return row


def summary_datasets():
    rows = []
    for dataset in DATASETS:
        rows.append(_dataset_summary(dataset))

    summary = pd.DataFrame(rows, columns=SUMMARY_COLS, index=DATASETS)
    print(summary)
    save_results(summary, 'data_summary')
    return
