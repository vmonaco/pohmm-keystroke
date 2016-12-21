## The Partially Observable Hidden Markov Model and its Application to Keystroke Biometrics

This repository contains code to reproduce experiments in this [article](http://arxiv.org/pdf/1607.03854.pdf).

### Dependencies

The code requires the [pohmm](https://github.com/vmonaco/pohmm) python package to be installed. Results were obtained with the following software versions:

    > %watermark -v -p numpy,scipy,pandas,sklearn,seaborn,pohmm
    CPython 3.5.1
    IPython 5.1.0
    
    numpy 1.10.4
    scipy 0.16.0
    pandas 0.18.0
    sklearn 0.17.1
    seaborn 0.7.0
    pohmm 0.2

It is recommended to use [Anaconda](https://www.continuum.io/downloads) and create a virtual env with the above dependencies installed.

### Steps to reproduce results

First, make sure the raw datasets are in the `data/raw` directory. Some datasets require permission to download. Then run the scripts below to preprocess the data and produce the main results. Some functions are very computationally intensive and may take several days to complete. 

For identification and verification results, run:

    $ python main_classify.py

For goodness of fit results, run:

    $ python main_montecarlo.py
    
For plots and postprocessing, run:

    $ python main_plots.py

For prediction results (not included in the article), run:
    
    $ python main_predict.py
