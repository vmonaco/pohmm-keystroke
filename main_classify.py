from pohmm_keystroke.data import preprocess_raw_data, summary_datasets
from pohmm_keystroke.classify import classification_results

if __name__ == '__main__':
    preprocess_raw_data()
    summary_datasets()
    classification_results()
