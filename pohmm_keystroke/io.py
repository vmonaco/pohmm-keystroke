import os
import sys
import pandas as pd
from . import DATA_DIR, RESULTS_DIR, FIGURES_DIR

try:
    from IPython.display import clear_output

    have_ipython = True
except ImportError:
    have_ipython = False


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.fill_char = '*'
        self.width = 40
        self.progress = 0
        print('_' * self.width, flush=True)
        sys.stdout.flush()

    def animate(self, iter):
        work_done = iter / float(self.iterations)
        len_progress = round(work_done * self.width)

        if len_progress > self.width:
            print('', flush=True)
        elif len_progress > self.progress:
            print(self.fill_char, end='')
            sys.stdout.flush()
            self.progress = len_progress


def exists_data(name):
    return os.path.exists(os.path.join(DATA_DIR, name + '.csv'))


def load_data(name, index_col=[0, 1], cols=None, **args):
    if cols:
        df = pd.read_csv(os.path.join(DATA_DIR, name + '.csv'),
                         index_col=index_col, usecols=list(index_col) + cols, **args)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, name + '.csv'), index_col=index_col, **args)

    return df


def save_data(df, name):
    df.to_csv(os.path.join(DATA_DIR, name + '.csv'))
    return


def load_results(name, index_col=0, **args):
    df = pd.read_csv(os.path.join(RESULTS_DIR, name + '.csv'), index_col=index_col, **args)
    return df


def save_results(df, name, subdir=''):
    df.to_csv(os.path.join(RESULTS_DIR, subdir, name + '.csv'))
    return


def load_txt(name):
    s = open(RESULTS_DIR + name + '.txt', 'rt').read()
    return s


def save_txt(text, name):
    with open(os.path.join(RESULTS_DIR, name + '.txt'), 'wt') as f:
        f.write(str(text))
    return

