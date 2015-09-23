# import relevant modules
import sys, os, re, argparse, itertools
import pandas, sklearn
import numpy as np
import scipy as sp
from waic import *

import theano.tensor as T
from sklearn.decomposition import PCA, SparsePCA, NMF
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.formula.api as smf
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sys import stdout
from collections import defaultdict
import patsy

np.random.seed(657824)

def load_data(fpath, value_name='similarity'):
    data = pandas.read_csv(fpath, sep=',')
    data.columns.values[0] = 'verb0'
    data = pandas.melt(data, id_vars=['verb0'], var_name='verb1', value_name=value_name)
    data = data[data.verb0 != data.verb1]

    return data
    
triad_data = load_data('model/triad_similarity_theano.csv', value_name='triadsim')
frame_data = load_data('model/likert_factor_analysis_theano/similarity/acceptability_FalseTrueTrue', value_name='likertsim')

sim_data = pandas.merge(triad_data, frame_data)

classification = pandas.read_csv('classifications/paper.csv', sep=',', index_col=0)

sim_data_all = pandas.merge(pandas.merge(sim_data, classification, left_on=['verb0'], right_index=True),
              classification, left_on=['verb1'], right_index=True)
            
regs = defaultdict(lambda: defaultdict(dict))
cors = [['var', 'predsparsity', 'cvverb', 'R2', 'loglike', 'spearman', 'pearson']]

#sim_data_all = pandas.concat([data, class_match], axis=1)

for var in ['triadsim', 'likertsim']:
    for predsparsity in [0., .1, .5, 1., 2., 5., 10.]:
        for verb in sim_data.verb0.unique():
            print verb, var
            train_ind = np.logical_and(sim_data.verb0!=verb, sim_data.verb1!=verb)
            test_ind = np.logical_not(train_ind)

            regs[var][predsparsity][verb] = smf.ols(formula=var+' ~ ('+'+'.join([col+'_x*'+col+'_y' for col in classification.columns])+')**1', data=sim_data_all[train_ind])

            if predsparsity:
                results = regs[var][predsparsity][verb].fit_regularized(alpha=predsparsity)
            else:
                results = regs[var][predsparsity][verb].fit()

            # cors.append([var, verb, predsparsity, sp.stats.spearmanr(sim_data_all[test_ind][var], results.predict(class_match[test_ind]))[0],
            #             sp.stats.pearsonr(sim_data_all[test_ind][var], results.predict(class_match[test_ind]))[0]])

            true_train = sim_data_all[train_ind][var]
            prediction_train = results.predict(sim_data_all[sim_data_all.columns[4:]][train_ind])

            resid_std = (true_train - prediction_train).std()

            true_test = sim_data_all[test_ind][var]
            prediction_test = results.predict(sim_data_all[sim_data_all.columns[4:]][test_ind])
            r2 = 1 - (true_test-prediction_test).var()/true_test.var()

            loglike = sp.stats.norm.logpdf(true_test-prediction_test, 0., resid_std).sum()

            cors.append([var, predsparsity, verb, r2, loglike, sp.stats.spearmanr(true_test, prediction_test)[0], sp.stats.pearsonr(true_test, prediction_test)[0]])
            
cors = pandas.DataFrame(cors[1:], columns=cors[0])

data = pandas.read_csv('model/likert_factor_analysis_theano/acceptability/acceptability_FalseTrueTrue', sep=',', index_col=0)

data_pca = (data - data.mean(axis=0))/np.sqrt(data.var(axis=0))
data_nmf = data-np.min(data.as_matrix())
data_nmf = data_nmf/np.max(data_nmf, axis=0)

fits = {}
design_matrices = defaultdict(dict)
regs_syntax = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
results_syntax = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

frame_cors = [['transform', 'predsparsity', 'cvverb', 'var', 'R2', 'loglike']]

for transform in ['pca', 'nmf']:
    for sparsity in [0., 0.2, .5, 1., 2., 5., 10.]:
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
        d.columns = ['c'*(col+1) for col in d.columns]
        
        data_all = pandas.merge(pandas.merge(sim_data, d, left_on='verb0', right_index=True),
                                d, left_on='verb1', right_index=True)
        # data_all1 = pandas.merge(sim_data, d, left_on='verb1', right_index=True)

        # data_all[d.columns] = data_all[d.columns] - data_all1[d.columns].as_matrix()
        
        for predsparsity in [0., .1, .5, 1., 2., 5., 10.]:
            for verb in data.index:
                train_ind = np.logical_and(sim_data.verb0!=verb, sim_data.verb1!=verb)
                test_ind = np.logical_not(train_ind)

                for var in ['triadsim', 'likertsim']:
                    print name, predsparsity, verb, var
                    formula_right = ' ~ '+'+'.join([col+'_x*'+col+'_y' for col in d.columns])
                    #formula_right = ' ~ '+'+'.join(d.columns)
                    formula = var+formula_right

                    _, design_matrices[name][var] = patsy.dmatrices(formula, data_all, return_type='dataframe')
                    
                    mod = regs_syntax[name][predsparsity][var][verb] = smf.ols(formula=formula, data=data_all[train_ind])

                    if predsparsity:
                        res = results_syntax[name][predsparsity][var][verb] = mod.fit_regularized(alpha=predsparsity)
                    else:
                        res = results_syntax[name][predsparsity][var][verb] = mod.fit()

                    true_train = data_all[var][train_ind]
                    predicted_train = res.predict(data_all[train_ind])
                    resid_std = (true_train - predicted_train).std()
                    
                    true = data_all[var][test_ind]
                    predicted = res.predict(data_all[test_ind])
        
                    r2 = 1 - (true-predicted).var()/true.var()

                    loglike = sp.stats.norm.logpdf(true-predicted, 0., resid_std).sum()
                    
                    frame_cors.append([name, predsparsity, verb, var, r2, loglike])

frame_cors = pandas.DataFrame(frame_cors[1:], columns=frame_cors[0])                    
frame_cors.groupby(['transform', 'predsparsity', 'var']).mean()

params_triad = []

for res in results_syntax['pca_sparse2.0'][0.0]['triadsim'].values():
    params_triad.append(np.array(res.params))

params_triad = np.array(params_triad)
params_triad = params_triad.mean(axis=0)

# sim_data['triadpredicted'] = params_triad[:,None] * design_matrices['pca_sparse2.0']['triadsim'].transpose()
triad_weighted = np.array(params_triad[:,None] * design_matrices['pca_sparse2.0']['triadsim'].transpose())
triad_weighted = triad_weighted[1::3] + triad_weighted[2::3] + triad_weighted[3::3]

triad_active = fits['pca_sparse2.0'].transform(data_pca)[:,triad_weighted.mean(axis=1) > 0]
triad_active = pandas.DataFrame(triad_active, index=data_pca.index)

triad_mapping = fits['pca_sparse2.0'].components_[triad_weighted.mean(axis=1) > 0].transpose()
triad_mapping = pandas.DataFrame(triad_mapping, index=data_pca.columns)

params_likert = []

for res in results_syntax['pca'][0.1]['triadsim'].values():
    params_likert.append(np.array(res.params))

params_likert = np.array(params_likert)
params_likert = params_likert.mean(axis=0)

# sim_data['likertpredicted'] = params_likert[:,None] * design_matrices['pca']['likertsim'].transpose()
likert_weighted = np.array(params_likert[:,None] * design_matrices['pca']['likertsim'].transpose())
likert_weighted = likert_weighted[1::3] + likert_weighted[2::3] + likert_weighted[3::3]

likert_active = fits['pca'].transform(data_pca)[:,likert_weighted.mean(axis=1) > 0]
likert_active = pandas.DataFrame(likert_active, index=data_pca.index)

likert_mapping = fits['pca'].components_[likert_weighted.mean(axis=1) > 0].transpose()
likert_mapping = pandas.DataFrame(likert_mapping, index=data_pca.columns)


# sim_data['triadresid'] = sim_data.triadpredicted + sim_data.triadsim - smf.ols(formula='triadsim - triadpredicted ~ likertsim', data=sim_data).fit().predict(sim_data)
# sim_data['likertresid'] = sim_data.likertpredicted + sim_data.likertsim - smf.ols(formula='likertsim - likertpredicted ~ triadsim', data=sim_data).fit().predict(sim_data)

# triad_resid = sim_data.pivot(index='verb0', columns='verb1', values='triadresid')
# likert_resid = sim_data.pivot(index='verb0', columns='verb1', values='likertresid')

# np.fill_diagonal(triad_resid.values, sim_data.triadresid.max())
# np.fill_diagonal(likert_resid.values, sim_data.likertresid.max())

# pandas.DataFrame(NMF(n_components=7, sparseness='data', max_iter=10000).fit_transform(triad_resid), index=triad_resid.index)
