# import relevant modules
import sys, os, re, argparse, itertools
import theano, pymc
import numpy as np
import scipy as sp

from sklearn.metrics import confusion_matrix 
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
# parser.add_argument('--frequency', 
#                     type=str, 
#                     default='../analysis/freq.csv')
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
parser.add_argument('--maptype', 
                    type=str, 
                    choices=['unweighted', 'weighted', 'interactive'], 
                    default='unweighted')

## parameter initialization
parser.add_argument('--loadmappings', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadjump', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadsubjsparsity', 
                    nargs='?',
                    const=True,
                    default=False)


## sampler parameters
parser.add_argument('--iterations', 
                    type=int, 
                    default=1100000)
parser.add_argument('--burnin', 
                    type=int, 
                    default=100000)
parser.add_argument('--thinning', 
                    type=int, 
                    default=1000)

## parse arguments
args = parser.parse_args()


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
                         skiprows=1)

## remove "know" from the likert data
know_bool = np.logical_and(likert_data[:,1] != 'know', likert_data[:,2] != 'know')
likert_data = likert_data[know_bool]

likert_subj_vals, likert_subj_indices = map_vals_to_indices(likert_data[:,0])
_, likert_verb1_indices = map_vals_to_indices(likert_data[:,1], vals=verbs)
_, likert_verb2_indices = map_vals_to_indices(likert_data[:,2], vals=verbs)
likert_responses = likert_data[:,3].astype(int) - 1

num_of_response_levels = np.unique(likert_responses).max() + 1
num_of_likert_subjects = likert_subj_vals.shape[0]
num_of_likert_observations = likert_data.shape[0]

## load triad data

triad_data = np.loadtxt(args.triaddata, 
                        delimiter=',', 
                        dtype=np.str, 
                        skiprows=1)

# subj, verb1, verb2, verb3, response_index

triad_subj_vals, triad_subj_indices = map_vals_to_indices(triad_data[:,0])
_, triad_verb1_indices = map_vals_to_indices(triad_data[:,1], vals=verbs)
_, triad_verb2_indices = map_vals_to_indices(triad_data[:,2], vals=verbs)
_, triad_verb3_indices = map_vals_to_indices(triad_data[:,3], vals=verbs)
triad_responses = triad_data[:,4].astype(int)

num_of_triad_subjects = triad_subj_vals.shape[0]
num_of_triad_observations = triad_data.shape[0]


## verb features

with open(args.verbfeatures) as f:
    verb_features_head = f.readline().strip().split(';')
    
try:
    assert np.all(np.array([verbs[i] == v for i, v in enumerate(verb_features_head)]))
except AssertionError:
    raise ValueError('Dude, the verbs need to be the same ones, in the same order')

verb_features = np.loadtxt(args.verbfeatures,
                           dtype=np.float,
                           delimiter=';',
                           skiprows=1).transpose()

num_of_features = verb_features.shape[1]

###########################
## initialization functions
###########################

def initialize_mapping(data):
    Tau = 1./num_of_features * np.identity(num_of_features)

    if args.loadmappings:
        if data == 'triad':
            return np.loadtxt(os.path.join(args.output, 'feature_weights_triad_'+args.maptype+'_all.csv'), 
                              delimiter=';', 
                              dtype=np.float, 
                              ndmin=2)

        else:
            return np.loadtxt(os.path.join(args.output, 'feature_weights_likert_'+args.maptype+'_all.csv'), 
                              delimiter=';', 
                              dtype=np.float, 
                              ndmin=2)
        
    else:
        return Tau


def initialize_jump():
    if args.loadjump:
        return np.loadtxt(os.path.join(args.output, 'jump_'+str(args.maptype)+'.csv'),
                          delimiter=';', 
                          dtype=np.float, 
                          skiprows=1).transpose()

    else:
        return sp.stats.expon.rvs(scale=1.,
                                  size=(num_of_subjects,
                                        num_of_response_levels-1))

def initialize_subj_sparsity():
    if args.loadsubjsparsity:
        return np.loadtxt(os.path.join(args.output, 'subj_sparsity_'+str(args.maptype)+'.csv'),
                          delimiter=';', 
                          dtype=np.float, 
                          skiprows=1).transpose()

    else:
        return sp.stats.expon.rvs(1., size=[num_of_triad_subjects, 3])


################
## mapping model
################

Tau = 1./num_of_features * np.identity(num_of_features)

if args.maptype=='interactive':
    feature_weights_triad = pymc.Wishart(name='feature_weights_triad',
                                         n=num_of_features,
                                         Tau=Tau,
                                         value=initialize_mapping('triad'),
                                         observed=False)

    feature_weights_likert = pymc.Wishart(name='feature_weights_likert',
                                          n=num_of_features,
                                          Tau=Tau,
                                          value=initialize_mapping('likert'),
                                          observed=False)

elif args.maptype=='weighted':
    feature_weights_triad_raw = pymc.Wishart(name='feature_weights_triad_raw',
                                             n=num_of_features,
                                             Tau=Tau,
                                             value=initialize_mapping('triad'),
                                             observed=False)

    feature_weights_likert_raw = pymc.Wishart(name='feature_weights_likert_raw',
                                              n=num_of_features,
                                              Tau=Tau,
                                              value=initialize_mapping('likert'),
                                              observed=False)

    @pymc.deterministic
    def feature_weights_triad(weights=feature_weights_triad_raw):
        weights_mat = np.zeros([num_of_features, num_of_features])
        np.fill_diagonal(weights_mat, np.diag(weights))

        return weights_mat

    @pymc.deterministic
    def feature_weights_likert(weights=feature_weights_likert_raw):
        weights_mat = np.zeros([num_of_features, num_of_features])
        np.fill_diagonal(weights_mat, np.diag(weights))

        return weights_mat

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
    sim = np.dot(np.dot(verb_features, w), verb_features.transpose())
    return sim - np.min(sim)

@pymc.deterministic
def similarity_likert(w=feature_weights_likert):
    sim = np.dot(np.dot(verb_features, w), verb_features.transpose())
    return sim - np.min(sim)

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
    cdfs = pymc.invlogit(cumsums - similarity[likert_verb1_indices, likert_verb2_indices, None])

    zeros = np.zeros(cdfs.shape[0])[:,None]
    ones = np.ones(cdfs.shape[0])[:,None]

    return np.append(cdfs, ones, axis=1) - np.append(zeros, cdfs, axis=1)


## odd man out model (triad similarity)

subj_sparsity_prior = pymc.Exponential(name='subj_sparsity_prior',
                                       beta=1.,
                                       value=sp.stats.expon.rvs(1., size=3),
                                       observed=False)

@pymc.deterministic(trace=False)
def subj_sparsity_prior_tile(subj_sparsity_prior=subj_sparsity_prior):
    return np.tile(subj_sparsity_prior, [num_of_triad_subjects, 1])

subj_sparsity = pymc.Exponential(name='subj_sparsity',
                                 beta=subj_sparsity_prior_tile,
                                 value=initialize_subj_sparsity(),
                                 observed=False)

@pymc.deterministic
def triad_params(sparsity=subj_sparsity, similarity=similarity_triad):
    return np.array([sparsity[triad_subj_indices, 0]*(similarity[triad_verb2_indices, triad_verb3_indices]+1e-10), 
                     sparsity[triad_subj_indices, 1]*(similarity[triad_verb1_indices, triad_verb3_indices]+1e-10), 
                     sparsity[triad_subj_indices, 2]*(similarity[triad_verb1_indices, triad_verb2_indices]+1e-10)]).transpose()

prob_triad = pymc.Dirichlet(name='prob_triad',
                            theta=triad_params,
                            value=np.random.dirichlet(alpha=[1., 1., 1.], 
                                                      size=num_of_triad_observations)[:,:2],
                            observed=False)

prob_triad_completed = pymc.Lambda(name='prob_triad_completed',
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

## get best kernel
kernel_triad_best = model.feature_weights_triad.trace()[minimum_index]
kernel_likert_best = model.feature_weights_likert.trace()[minimum_index]

## get best similarity
similarity_triad_best = model.similarity_triad.trace()[minimum_index]
similarity_likert_best = model.similarity_likert.trace()[minimum_index]

## get best random effects
jump_best = model.jump.trace()[minimum_index]
subj_sparsity_best = model.subj_sparsity.trace()[minimum_index]

##

triad_prediction = np.argmax(prob_triad_completed.trace()[minimum_index], axis=1)[triad_training_boolean]
likert_prediction = np.argmax(prob_likert.trace()[minimum_index], axis=1)[likert_training_boolean]

triad_real = triad_data[triad_training_boolean,4].astype(int)

triad_confusion = confusion_matrix(triad_data[triad_training_boolean,triad_real+1], 
                                   triad_data[triad_training_boolean,triad_prediction+1],
                                   labels=verbs)

##

if args.output:
    model_dir = args.output

    if args.testverb != 'all':
        model_dir = os.path.join(model_dir, 'crossvalidation')

        likert_test_boolean = np.logical_not(likert_training_boolean)
        triad_test_boolean = np.logical_not(triad_training_boolean)

        triad_test_ppd = prob_triad_completed.trace()[:, triad_test_boolean, triad_responses[triad_test_boolean]].mean(axis=0)
        likert_test_ppd = prob_likert.trace()[:, likert_test_boolean, likert_responses[likert_test_boolean]].mean(axis=0)

        triad_test_lppd = np.log(triad_test_ppd)
        likert_test_lppd = np.log(likert_test_ppd)

        triad_validation = np.vstack([np.repeat([args.maptype], triad_test_lppd.shape[0]), 
                                      np.repeat(['triad'], triad_test_lppd.shape[0]), 
                                      np.repeat([args.testverb], triad_test_lppd.shape[0]),
                                      np.array(verbs)[triad_verb1_indices[triad_test_boolean]], 
                                      np.array(verbs)[triad_verb2_indices[triad_test_boolean]], 
                                      np.array(verbs)[triad_verb3_indices[triad_test_boolean]],
                                      triad_test_lppd]).T
        likert_validation = np.vstack([np.repeat([args.maptype], likert_test_lppd.shape[0]), 
                                       np.repeat(['likert'], likert_test_lppd.shape[0]), 
                                       np.repeat([args.testverb], likert_test_lppd.shape[0]),
                                       np.array(verbs)[likert_verb1_indices[likert_test_boolean]], 
                                       np.array(verbs)[likert_verb2_indices[likert_test_boolean]], 
                                       np.array(verbs)[likert_verb2_indices[likert_test_boolean]], 
                                       likert_test_lppd]).T

        validation = np.vstack([triad_validation, likert_validation])

        with open(os.path.join(model_dir, 'test_lppd_'+args.maptype+'_'+args.testverb+'.csv'), 'a') as f:
            np.savetxt(f, validation, delimiter=';', fmt="%s")


    else:
        with open(os.path.join(args.output, 'waic'), 'a') as f:
            ## triad
            prob_trace = construct_prob_trace(trace=prob_triad_completed.trace(), responses=triad_responses)
            lppd = compute_lppd(prob_trace=prob_trace)
            waic = compute_waic(prob_trace=prob_trace)
        
            f.write(','.join([args.maptype, 'triad', str(lppd), str(waic)])+'\n')

            ## likert
            prob_trace = construct_prob_trace(trace=prob_likert.trace(), responses=likert_responses)
            lppd = compute_lppd(prob_trace=prob_trace)
            waic = compute_waic(prob_trace=prob_trace)
        
            f.write(','.join([args.maptype, 'likert', str(lppd), str(waic)])+'\n')

    np.savetxt(os.path.join(model_dir, 'prediction_triad_'+args.maptype+'_'+args.testverb+'.csv'), 
               triad_prediction,
               delimiter=';')
    np.savetxt(os.path.join(model_dir, 'prediction_likert_'+args.maptype+'_'+args.testverb+'.csv'), 
               likert_prediction,
               delimiter=';')

    np.savetxt(os.path.join(model_dir, 'feature_weights_triad_'+args.maptype+'_'+args.testverb+'.csv'), 
               kernel_triad_best,
               delimiter=';')
    np.savetxt(os.path.join(model_dir, 'feature_weights_likert_'+args.maptype+'_'+args.testverb+'.csv'), 
               kernel_likert_best,
               delimiter=';')

    np.savetxt(os.path.join(model_dir, 'similarity_triad_'+args.maptype+'_'+args.testverb+'.csv'), 
               similarity_triad_best,
               header=';'.join(verbs),
               delimiter=';',
               comments='')
    np.savetxt(os.path.join(model_dir, 'similarity_likert_'+args.maptype+'_'+args.testverb+'.csv'), 
               similarity_likert_best,
               header=';'.join(verbs),
               delimiter=';',
               comments='')

    np.savetxt(os.path.join(model_dir, 'confusion_triad_'+args.maptype+'_'+args.testverb+'.csv'), 
               triad_confusion,
               header=';'.join(verbs),
               delimiter=';',
               comments='')

    np.savetxt(os.path.join(model_dir, 'jump_'+str(args.maptype)+'.csv'), 
               jump_best.transpose(), 
               header=';'.join(likert_subj_vals),  
               delimiter=';',
               comments='')

    np.savetxt(os.path.join(model_dir, 'subj_sparsity_'+str(args.maptype)+'.csv'), 
               subj_sparsity_best.transpose(), 
               header=';'.join(triad_subj_vals),  
               delimiter=';',
               comments='')

    
