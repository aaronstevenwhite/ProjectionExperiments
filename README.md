# ProjectionExperiments

## Overview

This repository contains materials, data, and analysis scripts for one acceptability judgment experiment and two semantic similarity task investigating the relationship between propositional attitude verb syntax and semantics. These experiments were designed and run by [Aaron Steven White](http://aswhite.net), inconsultation with [Valentine Hacquard](http://ling.umd.edu/~hacquard) and [Jeffrey Lidz](http://ling.umd.edu/~jlidz) using [Alex Drummond](http://adrummond.net/)'s [Ibex](http://code.google.com/p/webspr/). The experiment was hosted on [Ibex Farm](http://spellout.net/ibexfarm/) and deployed on [Amazon Mechanical Turk](https://www.mturk.com/mturk/). 

A small portion of these data and analyses was presented at [NELS 43](https://nels2012.commons.gc.cuny.edu/) at CUNY. A prepublication version of the proceedings paper for that conference, "Discovering classes of attitude verbs using subcategorization frame distributions," can be found on Aaron Steven White's website. A journal-length paper entitled "Projecting Attitudes" is currently in preparation. A full draft of that paper is slated for release in mid- to late-April 2015.  

## Contents

### materials/

This directory contains all the materials needed to run each of the three experiments.

### data/

This directory contains directories for each experiment's data. Included in each directory are the raw data file pulled from Ibex (`*.ibex`), a file preprocessed using `preprocess.py` in the `analysis/` directory, and a file filtered using `filter.R` in the `analysis/` directory.

### analysis/

This directory contains preprocessing scripts (`preprocess.py`, `filter.R`), implementations of the two models presented in "Projecting Attitudes" (`likert_factor_analysis.py`, `similarity_model.py`), and an analysis script (`analysis.R`).
