# import relevant modules
import sys, os, re, argparse, itertools
import pandas, sklearn
import numpy as np
import scipy as sp
from waic import *

import theano.tensor as T
from sklearn.decomposition import PCA, SparsePCA, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sys import stdout
from collections import defaultdict

data = pandas.read_csv('model/likert_factor_analysis_theano/acceptability/acceptability_FalseTrueTrue', sep=',', index_col=0)

classification = pandas.read_csv('classifications/paper.csv', sep=',', index_col=0)

data_pca = (data - data.mean(axis=0))/np.sqrt(data.var(axis=0))
data_nmf = data-np.min(data.as_matrix())
data_nmf = data_nmf/np.max(data_nmf, axis=0)

fits = {}

svms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
coefs = np.array([np.append(['transformation', 'sparse', 'transformationsparsity', 'regularization', 'predictionsparsity',  'class', 'cvverb'], ['coef'+str(i) for i in range(data.shape[1])])])
weights = np.array([np.append(['transformation', 'sparse', 'transformationsparsity', 'regularization', 'predictionsparsity',  'class', 'cvverb'], data.columns)])
log_likes = np.array([['transformation', 'sparse', 'transformationsparsity', 'regularization', 'predictionsparsity',  'class', 'cvverb', 'trueclass', 'predictedclass', 'logprob', 'prob', 'baselineprob', 'accuracy', 'accuracyratio']])

np.random.seed(657824)

for transform in ['pca', 'nmf']:
    print transform
    for sparsity in [0., .1, .2, .5, 1., 2., 5., 10.]:
        print '\t', sparsity
        if transform in ['nmf', 'nmf_sparse']:
            if sparsity:
                transform = 'nmf_sparse'
                name = transform+str(sparsity)
                fits[name] = NMF(n_components=None, sparseness='data', max_iter=10000, beta=sparsity)
                fits[name].fit(data_nmf)
                d = fits[name].transform(data_nmf)
            else:
                transform = name = 'nmf'
                fits[name] = NMF(n_components=None, max_iter=10000)
                fits[name].fit(data_nmf)
                d = fits[name].transform(data_nmf)

        else:
            if sparsity:
                transform = 'pca_sparse' 
                name = transform+str(sparsity)
                fits[name] = SparsePCA(n_components=None, max_iter=10000, alpha=sparsity)
                fits[name].fit(data_pca)
                d = fits[name].transform(data_pca)
            else:
                transform = name = 'pca'
                fits[name] = PCA(n_components=None)
                fits[name].fit(data_pca)
                d = fits[name].transform(data_pca)
                            
        d = pandas.DataFrame(d, index=data.index)

        for cl in classification.columns:
            print '\t\t', cl
            classes = classification[cl].ix[d.index]
            baseline = classes.mean() if classes.mean() > 1 - classes.mean() else 1-classes.mean()
            if classes.sum() > 2:
                for verb in data.index:
                    for regularization in ['l1']:
                        for predsparsity in [.1, .2, .5, 1., 2., 5., 10.]:
                            mod = svms[name][cl][regularization][predsparsity][verb] = LogisticRegression(penalty=regularization, C=predsparsity).fit(d[d.index!=verb], classes[classes.index!=verb])
                            true_class = classes[verb]
                            predictors = d.ix[verb]
                            ll = np.log(mod.predict_proba(predictors)[0, true_class])
                            predicted_class = mod.predict(predictors)[0]
                            log_likes = np.append(log_likes, [[transform, sparsity==0, sparsity, regularization, 1./predsparsity, cl, verb, true_class, predicted_class, ll, np.exp(ll), baseline, float(np.exp(ll) > 0.5), float(np.exp(ll) > 0.5)/baseline]], axis=0)
                            coefs = np.append(coefs, [np.append([transform, sparsity==0, sparsity, regularization, 1./predsparsity, cl, verb], mod.coef_)], axis=0)
                            weights = np.append(weights, [np.append([transform, sparsity==0, sparsity, regularization, 1./predsparsity, cl, verb], np.dot(mod.coef_, fits[name].components_))], axis=0)

log_likes = pandas.DataFrame(log_likes[1:], columns=log_likes[0])
log_likes.logprob = log_likes.logprob.astype(float)
log_likes.prob = log_likes.prob.astype(float)
log_likes.accuracy = log_likes.accuracy.astype(float)
log_likes.accuracyratio = log_likes.accuracyratio.astype(float)

probs = pandas.pivot_table(log_likes, values='prob', columns=['transformation', 'class'], aggfunc=np.mean)
accuracies = pandas.DataFrame(pandas.pivot_table(log_likes, values='accuracy', index=['transformation', 'regularization', 'transformationsparsity', 'predictionsparsity', 'class'], aggfunc=np.mean))
accuracies.reset_index(inplace=True)

accuracies_max_avg = pandas.pivot_table(accuracies, values='accuracy', index=['transformation'], columns=['class'], aggfunc=lambda x: np.max(np.round(x, 3)))
accuracies_max_avg[accuracies_max_avg.columns[::-1]].ix[['pca', 'pca_sparse', 'nmf', 'nmf_sparse']]

accuracies_max = accuracies[accuracies.groupby(by=['class']).transform(lambda x: x==np.max(x)).accuracy]
accuracies_max = accuracies_max[accuracies_max.groupby(by=['class']).transform(lambda x: x==np.max(x)).transformationsparsity]
accuracies_max = accuracies_max[accuracies_max.groupby(by=['class']).transform(lambda x: x==np.max(x)).predictionsparsity]

coefs_best = pandas.merge(accuracies_max, pandas.DataFrame(coefs[1:], columns=coefs[0]))
coefs_best = pandas.melt(coefs_best, id_vars=list(coefs_best.columns[:8]))

weights_best = pandas.merge(accuracies_max, pandas.DataFrame(weights[1:], columns=weights[0]))
weights_best = pandas.melt(weights_best, id_vars=list(weights_best.columns[:8]))
    
weights_best['value'] = weights_best['value'].astype(float)
weights_best = pandas.pivot_table(weights_best, index=['class', 'variable'], values='value', aggfunc=np.mean)
weights_best = pandas.DataFrame(weights_best)
weights_best.reset_index(inplace=True)

weights_best.to_csv('model/frame_weights.csv')
