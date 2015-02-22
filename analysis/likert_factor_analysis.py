# import relevant modules
import sys, os, re, argparse, itertools
import theano, pymc
import numpy as np
import scipy as sp
from waic import *

## example: ipython -i -- likert_factor_analysis.py --loadverbfeatures --loadfeatureloadings --loadjump --loaditem --featurenum 5

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run likert factor analysis.')

## file handling
parser.add_argument('--verbs', 
                    type=str, 
                    default='../materials/triad/verbs.list')
parser.add_argument('--data', 
                    type=str, 
                    default='../data/frame/frame.filtered')
parser.add_argument('--output', 
                    type=str, 
                    default='./model/likert_factor_analysis')

## model hyperparameters
parser.add_argument('--featureform', 
                    type=str, 
                    choices=['discrete', 'continuous'], 
                    default='discrete')
parser.add_argument('--featurenum', 
                    type=int, 
                    default=10)
parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--loadingsparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--loadingprior', 
                    type=str, 
                    choices=['exponential', 'laplace'], 
                    default='exponential')
parser.add_argument('--nonparametric', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--includeitemrandom', 
                    nargs='?', 
                    const=True, 
                    default=False)

## parameter initialization
parser.add_argument('--loadverbfeatures', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadfeatureloadings', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadjump', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loaditem', 
                    nargs='?',
                    const=True,
                    default=False)

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

## parse arguments
args = parser.parse_args()

## filter bad parameter sets
if args.featureform == 'continuous' and args.nonparametric:
    raise NotImplementedError('hierarchical beta process feature prior not implemented')

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

verb_vals = [line.strip() for line in open(args.verbs)]

data = np.loadtxt(args.data, 
                  delimiter=';', 
                  dtype=np.str, 
                  skiprows=1)

## map to indices

subj_vals, subj_indices = map_vals_to_indices(data[:,0])
item_vals, item_indices = map_vals_to_indices(data[:,1].astype(int))
_, verb_indices = map_vals_to_indices(data[:,2], vals=verb_vals)
frame_vals, frame_indices = map_vals_to_indices(data[:,3])
responses = data[:,4].astype(int) - 1

num_of_verbs = len(verb_vals)
num_of_frames = frame_vals.shape[0]
num_of_subjects = subj_vals.shape[0]
num_of_items = item_vals.shape[0]
num_of_response_levels = np.unique(responses).shape[0]

###########################
## initialization functions
###########################

def initialize_verb_features():
    if args.loadverbfeatures:
        return np.loadtxt(os.path.join(args.output, 'verbfeatures_'+str(args.featurenum)+'.csv'), 
                          delimiter=';', 
                          dtype=np.int, 
                          skiprows=1,
                          ndmin=2).transpose()
    else:
        if args.featureform == 'discrete':
            return sp.stats.bernoulli.rvs(float(args.featuresparsity)/args.featurenum, 
                                          size=[num_of_verbs, args.featurenum])
        elif args.featureform == 'continuous':
            return sp.stats.beta.rvs(float(args.featuresparsity)/args.featurenum, 
                                     1., 
                                     size=[num_of_verbs, args.featurenum])
        

def initialize_feature_loadings():
    if args.loadfeatureloadings:
        feature_loadings = np.loadtxt(os.path.join(args.output, 'featureloadings_'+str(args.featurenum)+'.csv'), 
                                      delimiter=';', 
                                      dtype=np.float, 
                                      skiprows=1,
                                      ndmin=2)
            
    else:
        feature_loadings = sp.stats.expon.rvs(scale=1.,
                                              size=(args.featurenum,
                                                    num_of_frames))

    if args.loadingprior == 'exponential':
        return (feature_loadings,)
    else:
        return (feature_loadings/2, -feature_loadings/2)

def initialize_jump():
    if args.loadjump:
        return np.loadtxt(os.path.join(args.output, 'jump_'+str(args.featurenum)+'.csv'),
                          delimiter=';', 
                          dtype=np.float, 
                          skiprows=1).transpose()

    else:
        return sp.stats.expon.rvs(scale=1.,
                                  size=(num_of_subjects,
                                        num_of_response_levels-1))

def initialize_item():
    if args.loaditem:
        return np.loadtxt(os.path.join(args.output, 'item_'+str(args.featurenum)+'.csv'), 
                          delimiter=';', 
                          dtype=np.float, 
                          skiprows=1).transpose()

    else:
        return np.random.normal(0., 
                                item_prior.value,
                                size=num_of_items)


######################
## acceptability model
######################

## feature model

if args.featureform == 'discrete':
    feature_prob = pymc.Beta(name='feature_prob',
                             alpha=float(args.featuresparsity)/args.featurenum,
                             beta=1.,
                             value=sp.stats.beta.rvs(a=float(args.featuresparsity)/args.featurenum,
                                                     b=1., 
                                                     size=args.featurenum),
                             observed=False)

elif args.featureform == 'continuous':
    feature_prob = pymc.Exponential(name='feature_prob',
                                    beta=args.featurenum/float(args.featuresparsity),
                                    value=sp.stats.expon.rvs(args.featurenum/float(args.featuresparsity),
                                                             size=args.featurenum),
                                    observed=False)



if args.nonparametric:

    @pymc.deterministic(trace=False)
    def ibp_stick(feature_prob=feature_prob):
        probs = np.cumprod(feature_prob)

        return np.tile(probs, (num_of_verbs,1))


    verb_features = pymc.Bernoulli(name='verb_features',
                                   p=ibp_stick,
                                   value=initialize_verb_features(),
                                   observed=False)

else:
    @pymc.deterministic(trace=False)
    def feature_prob_tile(probs=feature_prob):
        return np.tile(probs, (num_of_verbs,1))

    if args.featureform == 'discrete':
        verb_features = pymc.Bernoulli(name='verb_features',
                                       p=feature_prob_tile,
                                       value=initialize_verb_features(),
                                       observed=False)
    
    elif args.featureform == 'continuous':
        verb_features = pymc.Beta(name='verb_features',
                                  alpha=feature_prob_tile,
                                  beta=np.ones([num_of_verbs, args.featurenum]),
                                  value=initialize_verb_features(),
                                  observed=False)


## projection model


if args.loadingprior == 'exponential':

    feature_loadings = pymc.Exponential(name='feature_loadings',
                                            beta=args.loadingsparsity,
                                            value=initialize_feature_loadings()[0],
                                            observed=False)

elif args.loadingprior == 'laplace':

    feature_loadings1 = pymc.Exponential(name='feature_loadings1',
                                         beta=1./args.loadingsparsity,
                                         value=initialize_feature_loadings()[0],
                                         observed=False,
                                         trace=False)

    feature_loadings2 = pymc.Exponential(name='feature_loadings2',
                                         beta=1./args.loadingsparsity,
                                         value=initialize_feature_loadings()[1],
                                         observed=False,
                                         trace=False)

    @pymc.deterministic
    def feature_loadings(fl1=feature_loadings1, fl2=feature_loadings2):
        return fl1 - fl2

## acceptability model

@pymc.deterministic(trace=False)
def acceptability(verb_features=verb_features, feature_loadings=feature_loadings):
    return np.dot(verb_features, feature_loadings)

#################
## response model
#################

## subject random effects

jump_prior = pymc.Gamma(name='jump_prior', 
                        alpha=0.001,
                        beta=0.001,
                        value=sp.stats.expon.rvs(scale=.1),
                        observed=False)


jump = pymc.Exponential(name='jump',
                        beta=jump_prior,
                        value=initialize_jump(),
                        observed=False)

## item random effects

if args.includeitemrandom:
    item_prior = pymc.Gamma(name='item_prior',
                            alpha=0.001,
                            beta=0.001,
                            value=sp.stats.expon.rvs(scale=.1),
                            observed=False)

    item = pymc.Normal(name='item',
                       mu=0.,
                       tau=item_prior,
                       value=initialize_item(),
                       observed=False)

    @pymc.deterministic(trace=False)
    def acceptability_item(acceptability=acceptability, item=item):
        return acceptability[verb_indices, frame_indices] + item[item_indices]

else:
    @pymc.deterministic(trace=False)
    def acceptability_item(acceptability=acceptability):
        return acceptability[verb_indices, frame_indices]
    

@pymc.deterministic
def prob_likert(acceptability_item=acceptability_item, jump=jump):

    if args.loadingprior == 'exponential':
        cumsums = np.cumsum(jump, axis=1)
    elif args.loadingprior == 'laplace':
        cumsums_uncentered = np.cumsum(jump, axis=1)
        cumsums = cumsums_uncentered - cumsums_uncentered[cumsums_uncentered.shape[0]/2]

    cdfs = pymc.invlogit(cumsums[subj_indices] - acceptability_item[:,None])

    zeros = np.zeros(cdfs.shape[0])[:,None]
    ones = np.ones(cdfs.shape[0])[:,None]

    return np.append(cdfs, ones, axis=1) - np.append(zeros, cdfs, axis=1)



response = pymc.Categorical(name='response',
                            p=prob_likert,
                            value=responses,
                            observed=True)

############
## fit model
############

## initialize model and begin sampler
model = pymc.MCMC(locals())
model.sample(iter=args.iterations, burn=args.burnin, thin=args.thinning)

## get deviance trace, minimum deviance, and index of minimum deviance
deviance_trace = model.trace('deviance')()
deviance_min = deviance_trace.min()
minimum_index = np.where(deviance_trace == deviance_min)[0][0]

## get best fixed effects
verb_features_best = model.verb_features.trace()[minimum_index]
feature_loadings_best = model.feature_loadings.trace()[minimum_index]

## get best random effects
jump_best = model.jump.trace()[minimum_index]

if args.includeitemrandom:
    item_best = model.item.trace()[minimum_index]

## write output
if args.output:
    np.savetxt(os.path.join(args.output,
                            args.featureform,
                            'verbfeatures_'+args.featureform+'_'+str(args.featurenum)+'.csv'), 
               verb_features_best.transpose(), 
               header=';'.join(verb_vals),
               delimiter=';',
               comments='',
               fmt="%d")
    np.savetxt(os.path.join(args.output, 
                            args.featureform,
                            'featureloadings_'+args.featureform+'_'+str(args.featurenum)+'.csv'), 
               feature_loadings_best, 
               header=';'.join(frame_vals),
               delimiter=';',
               comments='')
    np.savetxt(os.path.join(args.output,
                            args.featureform,
                            'jump_'+args.featureform+'_'+str(args.featurenum)+'.csv'), 
               jump_best.transpose(), 
               header=';'.join(subj_vals),  
               delimiter=';',
               comments='')

    if args.includeitemrandom:
        np.savetxt(os.path.join(args.output,
                                args.featureform,
                                'item_'+args.featureform+'_'+str(args.featurenum)+'.csv'), 
                   item_best[None,:], 
                   header=';'.join(item_vals.astype(str)),  
                   delimiter=';',
                   comments='')

    with open(os.path.join(args.output, 'waic_'+args.featureform), 'a') as f:
        prob_trace = construct_prob_trace(trace=prob_likert.trace(), responses=responses)
        lppd = compute_lppd(prob_trace=prob_trace)
        waic = compute_waic(prob_trace=prob_trace)

        f.write(','.join([str(args.featurenum), str(lppd), str(waic)])+'\n')
