from pohmm_keystroke.montecarlo import dataset_goodness_of_fit

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    dataset_goodness_of_fit(dataset, 'keyname', out_name='%s_pohmm_montecarlo' % dataset)
    dataset_goodness_of_fit(dataset, 'none', out_name='%s_hmm_montecarlo' % dataset)
