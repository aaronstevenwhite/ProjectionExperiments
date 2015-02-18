# import relevant modules
import sys, os, re, argparse, itertools
import theano, pymc
import numpy as np
import scipy as sp
from waic import *

## example: ipython -i -- similarity_model.py --loadverbfeatures --loadfeatureloadings --loadjump --loaditem --featurenum 5

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run similarity model.')

## file handling
parser.add_argument('--verbs', 
                    type=str, 
                    default='../materials/triad/verbs.list')
parser.add_argument('--testverb', 
                    type=str, 
                    default='all')
parser.add_argument('--triaddata', 
                    type=str, 
                    default='../data/triad/triad.filtered')
parser.add_argument('--likertdata', 
                    type=str, 
                    default='../data/likert/likert.filtered')
parser.add_argument('--verbfeatures', 
                    type=str, 
                    default='../analysis/model/likert_factor_analysis/verbfeatures_14.csv')
parser.add_argument('--output', 
                    type=str, 
                    default='./model/similarity_model')

## model hyperparameters
parser.add_argument('--featurenum', 
                    type=int, 
                    default=10)
parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--loadingsparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--maptype', 
                    type=str, 
                    choices=['unweighted', 'weighted', 'interactive'], 
                    default='unweighted')

## parameter initialization


## sampler parameters
parser.add_argument('--iterations', 
                    type=int, 
                    default=11000000)
parser.add_argument('--burnin', 
                    type=int, 
                    default=1000000)
parser.add_argument('--thinning', 
                    type=int, 
                    default=10000)


####################
## utility functions
####################

def map_vals_to_indices(col, vals=[]):
    vals = np.unique(col) if not vals else np.array(vals)
    index_mapper = np.vectorize(lambda x: np.where(vals == x))
    
    return vals, index_mapper(col)


#######
## data
#######

## load verb list
verbs = [line.strip() for line in open(args.verbs)]

num_of_verbs = len(verbs)

## load likert data
likert_data = np.loadtxt(args.likertdata, 
                         delimiter=',', 
                         dtype=np.str,
                         skiprows=1)[:,[0, 3, 4, 5]]

## remove "know" from the likert data
know_bool = np.logical_and(likert_data[:,1] != 'know', likert_data[:,2] != 'know')
likert_data = likert_data[know_bool]

likert_subj_vals, likert_subj_indices = map_vals_to_indices(likert_data[:,0])
_, likert_verb1_indices = map_vals_to_indices(likert_data[:,1], vals=verbs)
_, likert_verb2_indices = map_vals_to_indices(likert_data[:,2], vals=verbs)
likert_responses = likert_data[:,3].astype(int) - 1

num_of_response_levels = np.unique(responses).max() + 1
num_of_likert_subjects = likert_subj_vals.shape[0]
num_of_likert_observations = likert_data.shape[0]

## load triad data

triad_data = np.loadtxt(args.triaddata, 
                        delimiter=',', 
                        dtype=np.str, 
                        skiprows=1)[:,[0,3,4,5,6]]

# subj, verb1, verb2, verb3, response_index

triad_subj_vals, triad_subj_indices = map_vals_to_indices(triad_data[:,0])
_, triad_verb1_indices = map_vals_to_indices(triad_data[:,1], vals=verbs)
_, triad_verb2_indices = map_vals_to_indices(triad_data[:,2], vals=verbs)
_, triad_verb3_indices = map_vals_to_indices(triad_data[:,3], vals=verbs)
triad_responses = triad_data[:,4]

num_of_triad_subjects = triad_subj_vals.shape[0]
num_of_triad_observations = triad_data.shape[0]


## verb features

with open(args.verbfeatures) as f:
    verb_features_head = f.readline().strip().split()
    
try:
    assert np.all(np.array([verbs[i] == v for i, v in verb_features_head]))

verb_features = np.loadtxt(args.verbfeatures,
                           dtype=np.float,
                           skiprows=1).transpose()

num_of_features = verb_features.shape[1]


################
## mapping model
################

Tau = (1./num_of_features) * np.identity(num_of_features)

if args.maptype=='interactive':
    feature_weights_triad = pymc.Wishart(name='feature_weights_triad',
                                         n=num_of_features,
                                         Tau=Tau,
                                         value=Tau,
                                         observed=False)

    feature_weights_likert = pymc.Wishart(name='feature_weights_likert',
                                          n=num_of_features,
                                          Tau=Tau,
                                          value=Tau,
                                          observed=False)

elif args.maptype=='weighted':
     feature_weights_triad = Tau * pymc.Chi2(name='feature_weights_triad',
                                             n=num_of_features,
                                             value=num_of_features*np.ones(num_of_features),
                                             observed=False)

    feature_weights_likert = Tau * pymc.Chi2(name='feature_weights_likert',
                                             n=num_of_features,
                                             value=num_of_features*np.ones(num_of_features),
                                             observed=False)

elif args.maptype=='unweighted':
    @pymc.deterministic
    def feature_weights_triad():
        return Tau

    @pymc.deterministic
    def feature_weights_likert():
        return Tau


## similarities

@pymc.deterministic
def similarity_triad(w=feature_weights_triad):
    return np.dot(np.dot(verb_features, w), verb_features.transpose())

@pymc.deterministic
def similarity_likert(w=feature_weights_likert):
    return np.dot(np.dot(verb_features, w), verb_features.transpose())

## ordinal regression model (likert scale similarity)

jump_prior = pymc.Exponential(name='jump_prior', 
                              beta=1.,
                              value=sp.stats.expon.rvs(scale=.1),
                              observed=False)

jump = pymc.Exponential(name='jump', 
                        beta=jump_prior,
                        value=sp.stats.expon.rvs(scale=jump_prior.value,
                                                 size=[num_of_likert_subjects,
                                                       num_of_response_levels-1]),
                        observed=False)

@pymc.deterministic
def prob_likert(jump=jump, similarity=similarity_likert):
    cumsums = np.cumsum(jump, axis=1)[likert_subj_indices]
    cdfs = pymc.invlogit(cumsums - similarity[likert_verb1_indices, likert_verb2_indices])

    zeros = np.zeros(cdfs.shape[0])
    ones = np.ones(cdfs.shape[0])

    return np.append(cdfs, ones, axis=1) - np.append(zeros, cdfs, axis=1)


## odd man out model (triad similarity)

subj_sparsity_prior = pymc.Exponential(name='subj_sparsity_prior',
                                       beta=1.,
                                       value=sp.stats.expon.rvs(1., size=3),
                                       observed=False)

subj_sparsity = pymc.Exponential(name='subj_sparsity',
                                 beta=subj_sparsity_prior,
                                 value=sp.stats.expon.rvs(1., size=[num_of_triad_subjects, 3]),
                                 observed=False)

@pymc.deterministic
def triad_params(sparsity=subj_sparsity, similarity=similarity_triad):
    np.array([sparsity[subj_ind_triad, 0]*similarity[v4_ind, v5_ind], 
              sparsity[subj_ind_triad, 1]*similarity[v3_ind, v5_ind], 
              sparsity[subj_ind_triad, 2]*similarity[v3_ind, v4_ind]]).transpose()

prob_triad = pymc.Dirichlet(name='prob_triad',
                            theta=triad_params,
                            value=np.random.dirichlet(alpha=[1., 1., 1.], 
                                                      size=num_of_triad_observations)[:,:2],
                            observed=False)

prob_triad_completed = pymc.Lambda(name='prob_triad_completed_'+test_verb,
                                   lam_fun=lambda prob_triad=prob_triad: np.append(prob_triad, 1-np.sum(prob_triad, axis=1)[:,None], axis=1))


likert_training_boolean = np.logical_and(likert_data[:,1] != args.testverb, 
                                         likert_data[:,2] != args.testverb)
triad_training_boolean = np.logical_and(np.logical_and(triad_data[:,1] != args.testverb, 
                                                       triad_data[:,2] != args.testverb), 
                                        triad_data[:,3] != args.testverb)


likert = pymc.Categorical(name='likert',
                          p=prob_likert[likert_training_boolean],
                          value=likert_responses[likert_training_boolean],
                          observed=True)

triad = pymc.Categorical(name='triad',
                         p=prob_triad_completed[triad_training_boolean],
                         value=triad_responses[triad_training_boolean],
                         observed=True)

model = pymc.MCMC(locals())
model.sample(iter=args.iterations, burn=args.burnin, thin=args.thinning)

## write results

deviance_trace = model.trace('deviance')()
minimum_index = np.where(deviance_trace == deviance_trace.min())[0][0]

triad_prediction = np.argmax(prob_triad_completed.trace()[minimum_index], axis=1)
likert_prediction = np.argmax(prob_likert.trace()[minimum_index], axis=1)

kernel_triad_best = model.feature_weights_triad.trace()[minimum_index]
kernel_likert_best = model.feature_weights_likert.trace()[minimum_index]

similarity_triad_best = model.similarity_triad.trace()[minimum_index]
similarity_likert_best = model.similarity_likert.trace()[minimum_index]

np.savetxt('crossvalidation/prediction/prediction_triad_'+test_verb+'.csv', triad_prediction)
np.savetxt('crossvalidation/prediction/prediction_likert_'+test_verb+'.csv', likert_prediction)

np.savetxt('crossvalidation/feature_weights/feature_weights_triad_'+test_verb+'.csv', sp.linalg.cholesky(kernel_triad_best))
np.savetxt('crossvalidation/feature_weights/feature_weights_likert_'+test_verb+'.csv', sp.linalg.cholesky(kernel_likert_best))

np.savetxt('crossvalidation/similarity/similarity_triad_'+test_verb+'.csv', similarity_triad_best)
np.savetxt('crossvalidation/similarity/similarity_likert_'+test_verb+'.csv', similarity_likert_best)

##

likert_test_boolean = np.logical_not(likert_training_boolean)
triad_test_boolean = np.logical_not(triad_training_boolean)

prob_likert.trace()[:, likert_test_boolean, likert_responses[likert_test_boolean]]
prob_triad_completed.trace()[:, triad_test_boolean, triad_responses[triad_test_boolean]]
