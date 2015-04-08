# import relevant modules
import sys, os, re, argparse, itertools, copy
import theano, pymc
import numpy as np
import scipy as sp

from sklearn.metrics import confusion_matrix 
from waic import *

## example: ipython -i -- similarity_model.py --dataset triad --maptype weighted --kerneltype diffusion

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
parser.add_argument('--triaddata', 
                    type=str, 
                    default='../data/triad/triad.filtered')
parser.add_argument('--likertdata', 
                    type=str, 
                    default='../data/likert/likert.filtered')
parser.add_argument('--verbfeatures', 
                    type=str, 
                    default='../analysis/model/likert_factor_analysis/discrete/verbfeatures_14.csv')
parser.add_argument('--dataset', 
                    type=str, 
                    choices=['triad', 'likert', 'both'], 
                    default='both')
parser.add_argument('--output', 
                    type=str, 
                    default='./model/similarity_model')

## model hyperparameters
parser.add_argument('--maptype', 
                    type=str, 
                    choices=['unweighted', 'weighted'], 
                    default='unweighted')
parser.add_argument('--kerneltype', 
                    type=str, 
                    choices=['linear', 'diffusion'], 
                    default='linear')

## sampler parameters
parser.add_argument('--iterlim', 
                    type=int, 
                    default=1e10)
parser.add_argument('--repeats', 
                    type=int, 
                    default=100)

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
verbs = [line.strip() for line in open(args.verbs)] + ['know']

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

num_of_response_levels = np.unique(likert_responses).shape[0] # only works if every scale point was used at least once
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

def initialize_mapping():

    return np.random.exponential(1., num_of_features)

def initialize_jump():
    return sp.stats.expon.rvs(scale=1.,
                              size=(num_of_likert_subjects,
                                    num_of_response_levels-1))

def initialize_subj_bias():
    return sp.stats.expon.rvs(1., size=[num_of_triad_subjects, 3])


def construct_model(testverb=None):

    ################
    ## mapping model
    ################

    if args.maptype=='weighted':
        feature_weights = pymc.Gamma(name='feature_weights', 
                               alpha=0.001,
                               beta=0.001,
                               value=initialize_mapping(),
                               observed=False)

        if args.kerneltype == 'diffusion':

            bandwidth = pymc.Gamma(name='bandwidth', 
                                   alpha=0.001,
                                   beta=0.001,
                                   value=sp.stats.expon.rvs(scale=1.),
                                   observed=False)

            @pymc.deterministic
            def similarity(w=feature_weights, beta=bandwidth):
                dist = sp.spatial.distance.pdist(verb_features, 
                                                 'wminkowski', 
                                                 p=1, 
                                                 w=w)

                sim = np.power(np.tanh(beta), dist)
                sim_scaled = sim / np.max(sim)

                return sp.spatial.distance.squareform(sim_scaled)

        elif args.kerneltype == 'linear':

            @pymc.deterministic
            def similarity(w=feature_weights):
                dist = sp.spatial.distance.pdist(verb_features, 
                                                 'wminkowski', 
                                                 p=1, 
                                                 w=w)
                sim = w.sum() - dist
                sim_scaled = sim / np.max(sim)

                return sp.spatial.distance.squareform(sim_scaled)

    elif args.maptype=='unweighted':

        if args.kerneltype == 'diffusion':

            bandwidth = pymc.Gamma(name='bandwidth', 
                                   alpha=1.,
                                   beta=1.,
                                   value=sp.stats.expon.rvs(scale=1.),
                                   observed=False)

            @pymc.deterministic
            def similarity(beta=bandwidth):
                dist = sp.spatial.distance.pdist(verb_features, 
                                                 'minkowski', 
                                                 p=1)

                sim = np.power(np.tanh(beta), dist)
                sim_scaled = sim / np.max(sim)

                return sp.spatial.distance.squareform(sim_scaled)

        elif args.kerneltype == 'linear':

            @pymc.deterministic
            def similarity():
                dist = sp.spatial.distance.pdist(verb_features, 
                                                 'minkowski', 
                                                 p=1)

                sim = num_of_features - dist
                sim_scaled = sim / np.max(sim)

                return sp.spatial.distance.squareform(sim_scaled)

    ###############################
    ## multinomial regression model
    ###############################

    if args.dataset != 'likert':

        subj_bias_prior = pymc.Gamma(name='subj_bias_prior', 
                                     alpha=1.,
                                     beta=1.,
                                     value=sp.stats.expon.rvs(scale=1., size=3),
                                     observed=False)

        @pymc.deterministic(trace=False)
        def subj_bias_prior_tile(subj_bias_prior=subj_bias_prior):
            return np.tile(subj_bias_prior, [num_of_triad_subjects, 1])

        subj_bias = pymc.Exponential(name='subj_bias',
                                     beta=subj_bias_prior_tile,
                                     value=initialize_subj_bias(),
                                     observed=False)

        @pymc.deterministic
        def prob_triad(bias=subj_bias, similarity=similarity):
            raw_weights = bias[triad_subj_indices] + np.array([similarity[triad_verb2_indices, triad_verb3_indices], 
                                                               similarity[triad_verb1_indices, triad_verb3_indices], 
                                                               similarity[triad_verb1_indices, triad_verb2_indices]]).transpose()

            exp_weights = np.exp(raw_weights)

            return exp_weights / exp_weights.sum(axis=1)[:,None]


        # triad_training_boolean = np.logical_and(np.logical_and(triad_data[:,1] != testverb, 
        #                                                        triad_data[:,2] != testverb), 
        #                                         triad_data[:,3] != testverb)

        triad = pymc.Categorical(name='triad',
                                 p=prob_triad,#[triad_training_boolean],
                                 value=triad_responses,#[triad_training_boolean],
                                 observed=True)


    ###########################
    ## ordinal regression model
    ###########################

    if args.dataset != 'triad':

        jump_prior = pymc.Gamma(name='jump_prior', 
                                alpha=1.,
                                beta=1.,
                                value=sp.stats.expon.rvs(scale=.1),
                                observed=False)


        jump = pymc.Exponential(name='jump',
                                beta=jump_prior,
                                value=initialize_jump(),
                                observed=False)


        @pymc.deterministic
        def prob_likert(jump=jump, similarity=similarity):
            cumsums = np.cumsum(jump, axis=1)[likert_subj_indices]

            cdfs = pymc.invlogit(cumsums-similarity[likert_verb1_indices, likert_verb2_indices, None])

            zeros = np.zeros(cdfs.shape[0])[:,None]
            ones = np.ones(cdfs.shape[0])[:,None]

            return np.append(cdfs, ones, axis=1) - np.append(zeros, cdfs, axis=1)


        # likert_training_boolean = np.logical_and(likert_data[:,1] != testverb, 
        #                                          likert_data[:,2] != testverb)

        likert = pymc.Categorical(name='likert',
                                  p=prob_likert,#[likert_training_boolean],
                                  value=likert_responses,#[likert_training_boolean],
                                  observed=True)    

    return locals()

####################
## fitting functions
####################

def compute_logprob(model):
    logprob = 0

    if args.dataset != 'triad':
        likert_probs = model.prob_likert.value
        likert_probs = likert_probs[range(likert_probs.shape[0]),likert_responses]

        logprob += np.log(likert_probs).sum()

    if args.dataset != 'likert':
        triad_probs = model.prob_triad.value
        triad_probs = triad_probs[range(triad_probs.shape[0]),triad_responses]

        logprob += np.log(triad_probs).sum()

    return logprob

def write_params(model, testverb='all'):
        np.savetxt(os.path.join(args.output, 'similarity_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                   model.similarity.value,
                   header=';'.join(verbs),
                   delimiter=';',
                   comments='')

        if args.maptype == 'weighted':
            np.savetxt(os.path.join(args.output, 'feature_weights_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                       model.feature_weights.value,
                       delimiter=';')

        if args.kerneltype == 'diffusion':
            with open(os.path.join(args.output, 'bandwidth_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 'w') as f:
                f.write(str(model.bandwidth.value))

        if args.dataset != 'likert':
            np.savetxt(os.path.join(args.output, 'subj_bias_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                       model.subj_bias.value.transpose(), 
                       header=';'.join(triad_subj_vals),  
                       delimiter=';',
                       comments='')

        if args.dataset != 'triad':
            np.savetxt(os.path.join(args.output, 'jump_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                       model.jump.value.transpose(), 
                       header=';'.join(likert_subj_vals),  
                       delimiter=';',
                       comments='')    


def write_confusion(model, testverb='all'):
    if args.dataset != 'likert':
        triad_prediction = np.argmax(model.prob_triad.value, axis=1)

        triad_confusion = confusion_matrix(triad_data[range(triad_responses.shape[0]),triad_responses+1], 
                                           triad_data[range(triad_responses.shape[0]),triad_prediction+1],
                                           labels=verbs[:30])

        np.savetxt(os.path.join(args.output, 'triad_confusion_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                   triad_confusion,
                   header=';'.join(verbs[:30]),
                   delimiter=';',
                   comments='',
                   fmt="%s")


    if args.dataset != 'triad':
        likert_probs = model.prob_likert.value
        likert_probs = likert_probs[range(likert_probs.shape[0]),likert_responses]

        likert_confusion = np.vstack([likert_data[:,[1,2]].T, likert_probs.astype(str)]).T

        np.savetxt(os.path.join(args.output, 'likert_confusion_'+args.dataset+'_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 
                   likert_confusion,
                   delimiter=';',
                   fmt="%s")
        

def write_fit_statistics(model):
    logprob = compute_logprob(model)
    
    fname = os.path.join(args.output, 'fitstats.csv')

    line = ','.join([args.dataset, args.maptype, args.kerneltype, 
                     str(-2*logprob), str(model.AIC), str(model.BIC)])

    with open(fname, 'a') as f:    
        f.write(line+'\n')

    return logprob

###############################
## fit model and cross validate
###############################

best_logprob = -np.inf

for i in range(args.repeats):
    print 'repeat:', i

    model = pymc.MAP(construct_model())
    model.fit(method="fmin_powell", iterlim=args.iterlim)

    curr_logprob = write_fit_statistics(model)

    if curr_logprob > best_logprob:
        best_logprob = curr_logprob

        write_params(model)
        write_confusion(model)

# crossvalidation = {}

# for v in verbs:
#     print v

#     print vardict['feature_weights'].value

#     vardict_cross = copy.copy(vardict)
#     vardict_cross.update(construct_observed(v))

#     model = pymc.MAP(vardict)
#     model.fit(iterlim=args.iterlim, verbose=True, method="fmin_powell")

#     crossvalidation[v] = vardict_cross

################
## write results
################

# if args.dataset != 'likert':
#     subj_bias_best = model.subj_bias.value

# if args.dataset != 'triad':
#     jump_best = model.jump.value
#     likert_prediction = np.argmax(model.prob_likert.value, axis=1)[likert_training_boolean]

# ##

# if args.output:


#     if testverb != 'all':
#         model_dir = os.path.join(args.output, 'crossvalidation')

#         likert_test_boolean = np.logical_not(likert_training_boolean)
#         triad_test_boolean = np.logical_not(triad_training_boolean)

#         triad_test_ppd = prob_triad_completed.trace()[:, triad_test_boolean, triad_responses[triad_test_boolean]].mean(axis=0)
#         likert_test_ppd = prob_likert.trace()[:, likert_test_boolean, likert_responses[likert_test_boolean]].mean(axis=0)

#         triad_test_lppd = np.log(triad_test_ppd)
#         likert_test_lppd = np.log(likert_test_ppd)

#         triad_validation = np.vstack([np.repeat([args.maptype], triad_test_lppd.shape[0]), 
#                                       np.repeat(['triad'], triad_test_lppd.shape[0]), 
#                                       np.repeat([testverb], triad_test_lppd.shape[0]),
#                                       np.array(verbs)[triad_verb1_indices[triad_test_boolean]], 
#                                       np.array(verbs)[triad_verb2_indices[triad_test_boolean]], 
#                                       np.array(verbs)[triad_verb3_indices[triad_test_boolean]],
#                                       triad_test_lppd]).T
#         likert_validation = np.vstack([np.repeat([args.maptype], likert_test_lppd.shape[0]), 
#                                        np.repeat(['likert'], likert_test_lppd.shape[0]), 
#                                        np.repeat([testverb], likert_test_lppd.shape[0]),
#                                        np.array(verbs)[likert_verb1_indices[likert_test_boolean]], 
#                                        np.array(verbs)[likert_verb2_indices[likert_test_boolean]], 
#                                        np.array(verbs)[likert_verb2_indices[likert_test_boolean]], 
#                                        likert_test_lppd]).T

#         validation = np.vstack([triad_validation, likert_validation])

#         with open(os.path.join(model_dir, 'test_lppd_'+args.maptype+'_'+args.kerneltype+'_'+testverb+'.csv'), 'a') as f:
#             np.savetxt(f, validation, delimiter=';', fmt="%s")


