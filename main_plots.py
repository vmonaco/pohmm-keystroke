from pohmm_keystroke.classify import plot_pohmm_example, plot_montecarlo_hmm_vs_pohmm, \
    plot_roc_curves_hmm_vs_pohmm, summary_table

if __name__ == '__main__':
    plot_pohmm_example('fixed_text')
    plot_pohmm_example('free_text')
    plot_montecarlo_hmm_vs_pohmm('fixed_text')
    plot_montecarlo_hmm_vs_pohmm('free_text')
    plot_roc_curves_hmm_vs_pohmm('password')
    plot_roc_curves_hmm_vs_pohmm('mobile')
    plot_roc_curves_hmm_vs_pohmm('keypad')
    plot_roc_curves_hmm_vs_pohmm('fixed_text')
    plot_roc_curves_hmm_vs_pohmm('free_text')
    summary_table('U-ACC')
    summary_table('U-EER')
    summary_table('AMRT')
