# import relevant modules
import sys, os, re, argparse, itertools
import theano, pandas, sklearn
import numpy as np
import scipy as sp
from waic import *

import theano.tensor as T

from sys import stdout
from collections import defaultdict

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run likert factor analysis.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../data/frame/frame.filtered')
parser.add_argument('--output', 
                    type=str, 
                    default='./model/likert_factor_analysis_theano')

## model hyperparameters
parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=-1) ## uniform(0,1)
parser.add_argument('--projectionsparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--jumpsparsity', 
                    type=float, 
                    default=1.)
# parser.add_argument('--loadingprior', 
#                     type=str, 
#                     choices=['exponential', 'laplace'], 
#                     default='exponential')

## optimizer parameters
parser.add_argument('--learningrate', 
                    type=float, 
                    default=.002)
parser.add_argument('--evidencesample', 
                    type=int, 
                    default=int(1e6))
parser.add_argument('--minibatchsize', 
                    type=int, 
                    default=7740) ## full dataset
parser.add_argument('--maxiter', 
                    type=int, 
                    default=int(1e6))
parser.add_argument('--momentum', 
                    type=float, 
                    default=0.9)
parser.add_argument('--annealing', 
                    type=int, 
                    default=int(1e5))
parser.add_argument('--tolerance', 
                    type=float, 
                    default=0.01)

## parse arguments
args = parser.parse_args()

#######################
## utility functions ##
#######################

def logistic(logodds, module=T):
    return 1 / (1 + module.exp(-logodds))

np.pow = np.power
T.abs = T.abs_

#######################
## class definitions ##
#######################

class Data(object):

    def __init__(self, fpath, cols=['verb', 'frame'], sep=';', return_shared=True):
        self._data = pandas.read_csv(fpath, sep=sep)
        self._cols = cols

        if 'item' not in self._data.columns:
            self._data['item'] = 0
        
        self.return_shared = return_shared
        
        self._shuffle_data()
        self._recast_columns()

    def __getitem__(self, item):
        return self.__dict__[item]
        
    def _shuffle_data(self):
        '''randomize data for stochastic gradient ascent'''
        self._data = self._data.reindex(np.random.permutation(self._data.index))
        
    def _recast_columns(self):
        '''
        cast all columns except response and applicable to category type
        (category type is the pandas version of R's factor)
        and create a theano shared variable for each
        '''
        
        for col in self._data:
            if col != 'response':
                self._data[col] = self._data[col].astype('category')

                colcodes = self._data[col].cat.codes.as_matrix()
                
                self.__dict__['_'+col] = theano.shared(colcodes, name=col)
                
            else:
                colcodes = self._data[col].as_matrix()
                self.__dict__['_'+col] = theano.shared(colcodes, name=col)

            self.__dict__[col] = self.__dict__['_'+col] if self.return_shared else colcodes

    def nlevels(self, col):
        '''Get the number of unique levels of a factor'''
        return self._data[col].cat.categories.shape[0]

    @property
    def ndatapoints(self):
        return self._data.shape[0]

    @property
    def maxresponse(self):
        return self._data.response.max()

    @property
    def minresponse(self):
        return self._data.response.min()
    
    @property
    def mean_ratings(self):
        return pandas.pivot_table(self._data,
                                  values='response',
                                  index=self._cols[0],
                                  columns=self._cols[1],
                                  aggfunc=np.mean, fill_value=0.).as_matrix()
    
    def reset_attributes(self, return_shared):
        self.return_shared = return_shared
        self._recast_columns()


class OrdinalRegression(object):

    def __init__(self, data, cols, symmetrize, equidistant, additive_subj_effects, multiplicative_subj_effects, **kwargs):

        self._data = data
        self._cols = cols

        self.symmetrize = symmetrize
        
        if kwargs is not None:
            self.__dict__.update(kwargs)

        self.equidistant = equidistant
        self.additive_subj_effects = additive_subj_effects
        self.multiplicative_subj_effects = multiplicative_subj_effects

        if 'num_of_features' in self.__dict__:
            self.tag = str(self.num_of_features)
        else:
            self.tag = str(self.equidistant) +\
                  str(self.additive_subj_effects) +\
                  str(self.multiplicative_subj_effects)
                    
        self._initalize_variables()        
        self._initialize_objective()

    def __getitem__(self, item):
        return self.__dict__[item]
        
    def _initalize_variables(self):
        self._variables = [] # list of strings giving variable names
        
        self._initialize_predictors()
        self._initialize_jumps()
        self._initialize_random_effects()

        self.indices = np.arange(self.ndatapoints).astype(np.int32)

    def _initialize_predictors(self):

        acceptability = self._data.mean_ratings
        self.acceptability = acceptability
        
        self._variables.extend(['acceptability'])

    def _initialize_jumps(self):

        percentile_func = lambda j: np.percentile(self.acceptability,
                                                  100*float(j)/self._data.maxresponse + 50/self._data.maxresponse)
        jump_func = lambda j: percentile_func(j) - percentile_func(j-1) 
        jumps = np.array([jump_func(i) for i in range(1, self._data.maxresponse)])

        if self.equidistant:
            self.jumps = np.mean(jumps)
        else:
            self.jumps = np.append(0., np.append(jumps, 0.))
            
        self._variables.extend(['jumps'])

    def _initialize_random_effects(self):

        self.item_stdev = 1.
        self.subj_add_stdev = 1.
        self.subj_mult_stdev = 1.

        self.item = np.ones(self._data.nlevels('item'))
        self.subj_add = np.zeros(self._data.nlevels('subj'))
        self.subj_mult = np.ones(self._data.nlevels('subj'))
        
        self._variables.extend(['item_stdev', 'subj_add_stdev', 'subj_mult_stdev',
                                'item', 'subj_add', 'subj_mult'])

    def _wrap_in_shared(self):
        
        for var in self._variables:
            self.__dict__[var] = theano.shared(self.__dict__[var], name=var+'_'+self.tag)

        self.indices = T.ivector('indices')


    def _initialize_objective(self):

        if self._data.return_shared:
            self._wrap_in_shared()
            module = T
        else:
            module = np

        self._initialize_predictors_prior(module)
        self._initialize_jumps_prior(module)
        self._initialize_ranef_prior(module)
        
        self._initialize_likelihood(module)
        
        ## prior probability
        self.log_prior = self.predictors_prior + self.jumps_prior + self.random_prior
                           
        ## posterior probability
        self.log_posterior = self.log_likelihood + self.log_prior

    def _initialize_predictors_prior(self, m):
        #self.predictors_prior = -m.sum(self.acceptability)
        self.predictors_prior = 0.

    def _initialize_jumps_prior(self, m):
        self.jumps_prior = 0.#-self._jump_sparsity * m.sum(self.jumps)

    def _initialize_ranef_prior(self, m):
        item_stdev = m.abs(self.item_stdev)
        self.item_prior = m.sum(-m.log(item_stdev) -\
                                         m.pow(self.item, 2) /\
                                         (2*m.pow(item_stdev, 2)))


        ## super hacky (and probably slow) way of getting these into the computational graph
        ## when we don't need them
        subj_add_stdev = m.abs(self.subj_add_stdev)
        self.subj_add_prior =T.switch(int(self.multiplicative_subj_effects),
                                       m.sum(-m.log(subj_add_stdev) -\
                                              m.pow(self.subj_add, 2) /\
                                              (2*m.pow(subj_add_stdev, 2))),
                                       subj_add_stdev-subj_add_stdev)         

        ## super hacky (and probably slow) way of getting these into the computational graph
        ## when we don't need them                                     
        subj_mult_stdev = m.abs(self.subj_mult_stdev)
        self.subj_mult_prior =T.switch(int(self.additive_subj_effects),
                                       m.sum(-m.log(subj_mult_stdev) -\
                                              m.pow(self.subj_mult, 2) /\
                                              (2*m.pow(subj_mult_stdev, 2))),
                                       subj_mult_stdev-subj_mult_stdev)         

        self.random_prior = self.item_prior + self.subj_add_prior + self.subj_mult_prior
                                    
    def _initialize_likelihood(self, m):

        if self.symmetrize:
            param = self.acceptability[self._data[self._cols[0]][self.indices],
                                       self._data[self._cols[1]][self.indices]] +\
                    self.acceptability[self._data[self._cols[1]][self.indices],
                                       self._data[self._cols[0]][self.indices]] +\
                    self.item[self._data.item[self.indices]]

        else:
            param = self.acceptability[self._data[self._cols[0]][self.indices],
                                       self._data[self._cols[1]][self.indices]] +\
                    self.item[self._data.item[self.indices]]
                
        response = self._data.response[self.indices]

        if self.equidistant:
            cutpoints = m.abs(self.jumps)*np.append(0., np.append(np.arange(1, self._data.maxresponse), 0.))
        else:
            ## even if jumps are negative, treated as positive because of abs;
            ## thus, well-ordered cutpoints
            cutpoints = m.cumsum(m.abs(self.jumps))

        ## super hacky (and probably slow) way of getting these into the computational graph
        ## when we don't need them
        param = T.switch(int(self.additive_subj_effects),
                         param + self.subj_add[self._data.subj[self.indices]],
                         param + self.subj_add[self._data.subj[self.indices]] - self.subj_add[self._data.subj[self.indices]])

        subj_mult = T.switch(int(self.multiplicative_subj_effects),
                             m.abs(self.subj_mult[self._data.subj[self.indices]]),
                             self.subj_mult[self._data.subj[self.indices]]/self.subj_mult[self._data.subj[self.indices]])

        if m is T:
            logistic_high = logistic(subj_mult*cutpoints[response] - param, m)
            logistic_low = logistic(subj_mult*cutpoints[response-1] - param, m)

            high = T.switch(T.lt(response, self._data.maxresponse),
                            logistic_high,
                            response-response+1.)
            low = T.switch(T.gt(response, self._data.minresponse),
                           logistic_low,
                           response-response+0.)
            
            self.prob = prob = high - low            
        else:
            cutpoints = np.append(-np.inf, np.append(cutpoints[1:-1], np.inf))

            if self.multiplicative_subj_effects:
                prob = logistic(subj_mult*cutpoints[response] - param, m) -\
                       logistic(subj_mult*cutpoints[response-1] - param, m)
            else:
                prob = logistic(cutpoints[response] - param, m) -\
                       logistic(cutpoints[response-1] - param, m)    
                                   

        self.log_likelihood = m.sum(m.log(prob))

        if self._data.return_shared:
            self._compute_log_likelihood = theano.function(inputs=[self.indices],
                                                           outputs=self.log_likelihood)

    def compute_log_likelihood(self):
        if self._data.return_shared:
            indices = np.arange(self.ndatapoints).astype(np.int32)
            return self._compute_log_likelihood(indices)

        else:
            return self.log_likelihood            

    def reinitialize_objective(self):
        self._initialize_objective()
        
    @property
    def ndatapoints(self):
        return self._data.ndatapoints

    @property
    def variables(self):
        return self._variables


    def get_as_df(self, var):

        if var == 'features':
            df = pandas.DataFrame(self.features.get_value(),
                                  index=self._data._data[self._cols[0]].cat.categories)

        elif var == 'projection':
            df = pandas.DataFrame(self.projection.get_value().transpose(),
                                  index=self._data._data[self._cols[1]].cat.categories)

        elif var == 'acceptability':
            if self.symmetrize:
                df = pandas.DataFrame(self.acceptability.eval()+self.acceptability.eval().transpose(),
                                      index=self._data._data[self._cols[0]].cat.categories,
                                      columns=self._data._data[self._cols[1]].cat.categories)
            else:
                df = pandas.DataFrame(self.acceptability.eval(),
                                      index=self._data._data[self._cols[0]].cat.categories,
                                      columns=self._data._data[self._cols[1]].cat.categories)

                                
        elif var == 'jumps':
            if self.equidistant:
                df = pandas.DataFrame([self.jumps.get_value()])
            else:
                df = pandas.DataFrame(self.jumps.get_value()[1:-1])

        elif var == 'item_stdev':
            df = pandas.DataFrame([self.item_stdev.get_value()])

        elif var == 'subj_add_stdev':
            df = pandas.DataFrame([self.subj_add_stdev.get_value()])

        elif var == 'subj_mult_stdev':
            df = pandas.DataFrame([self.subj_mult_stdev.get_value()])

            
        elif var == 'item':
            df = pandas.DataFrame(self.item.get_value(),
                                  index=self._data._data.item.cat.categories)

        elif var == 'subj_add':
            df = pandas.DataFrame(self.subj_add.get_value(),
                                  index=self._data._data.subj.cat.categories)

        elif var == 'subj_mult':
            df = pandas.DataFrame(self.subj_mult.get_value(),
                                  index=self._data._data.subj.cat.categories)

                        
        return df

    def write(self, output):
        
        for var in self._variables:                
            self.get_as_df(var).to_csv(os.path.join(output, var+'_'+self.tag))

            
        
class OrdinalFactorAnalysis(OrdinalRegression):

    def __init__(self, data, cols, num_of_features, feature_sparsity, projection_sparsity):

        super(OrdinalFactorAnalysis, self).__init__(data=data,
                                                    cols=cols,
                                                    num_of_features=num_of_features,
                                                    feature_sparsity=feature_sparsity,
                                                    projection_sparsity=projection_sparsity)
            
    def _initialize_predictors(self):

        if self.feature_sparsity == 0:
            self.feature_sparsity = np.random.exponential(1.)

        elif self.feature_sparsity == -1:
            self.feature_sparsity = np.random.uniform(0., 1.)

                        
        self.features = np.random.beta(self.feature_sparsity, self.feature_sparsity,
                                       size=[self._data.nlevels('verb'),
                                             self.num_of_features])

        if self.projection_sparsity == 0:
            self.projection_sparsity = np.random.exponential(1.)
        
        self.projection = np.random.exponential(self.projection_sparsity,
                                                size=[self.num_of_features,
                                                      self._data.nlevels('frame')])            

        self.acceptability = np.dot(self.features, self.projection)
        
        self._variables.extend(['features', 'projection'])


    def _wrap_in_shared(self):
        super(OrdinalFactorAnalysis, self)._wrap_in_shared()

        self.feature_sparsity = theano.shared(self.feature_sparsity,
                                              name='feature_sparsity_'+str(self.num_of_features))
        self.projection_sparsity = theano.shared(self.projection_sparsity,
                                              name='projection_sparsity_'+str(self.num_of_features))
            

    def _initialize_predictors_prior(self, m):

        self.feature_prior = m.sum((self.feature_sparsity-1) * m.log(self.features+1e-63) +\
                                   (self.feature_sparsity-1) * m.log(1 - self.features+1e-63))

        self.projection_prior = -self.projection_sparsity * m.sum(m.abs(self.projection))

        self.predictors_prior = self.feature_prior + self.projection_prior

        self.acceptability = m.dot(self.features, m.abs(self.projection)) ## this is what makes the projection nonnegative


                                
        
###############
## gradients ##
###############

class StochasticGradientAscent(object):

    def __init__(self, model, learning_rate_default, learning_rate_dict={}, mle=False):
        self._model = model
        self.mle = mle

        learning_rate_dict = defaultdict(lambda: learning_rate_default, learning_rate_dict)
        
        for lr in ['learning_rate_'+var for var in model.variables]:
            self.__dict__[lr] = learning_rate_dict[lr]
        
        self._initialize_updates()
        
    def _initialize_updates(self):

        #if self.mle:
        #    objective = self._model.log_likelihood
        #else:
        objective = self._model.log_posterior

        updates = []

        momentum = T.dscalar('momentum')
        iteration = T.dscalar('iteration')
        annealing = T.dscalar('annealing')
        
        for var in self._model.variables:

            self.__dict__[var+'_step'] = theano.shared(self._model[var].get_value()*0., name=var+'_step')
                                         
            updates.append((self._model[var], self._model[var]+\
                                              self.__dict__['learning_rate_'+var]*\
                                              self.__dict__[var+'_step']/(1+iteration/annealing)))
            updates.append((self.__dict__[var+'_step'], momentum*self.__dict__[var+'_step'] + (1.-momentum)*T.grad(objective, self._model[var])))
            
        self._update_vars = theano.function(inputs=[self._model.indices, momentum, iteration, annealing],
                                           outputs=[objective],
                                           updates=theano.compat.python2x.OrderedDict(updates),
                                           name='update_vars')


    def fit(self, minibatchsize, maxiter, tolerance, momentum=.9, annealing=2500, window_size=2500):

        self.post_hist = np.array([])
        self.like_hist = np.array([])
        self.prior_hist = np.array([])
        
        ## tolerance not used
        for i in range(maxiter):
            j = np.random.choice(a=self._model.ndatapoints,
                                 size=minibatchsize,
                                 replace=False).astype(np.int32)

            self.post_hist = np.append(self.post_hist, self._update_vars(j, momentum, i, annealing)[0])
            self.like_hist = np.append(self.like_hist, self._model.compute_log_likelihood())
            self.prior_hist = np.append(self.prior_hist, self.post_hist[-1] - self.like_hist[-1] )
            
            stdout.write('\r'+str(round(self.like_hist[-1],3))+'    '+str(round(self.like_hist[i-np.min([window_size,i])], 3))+'    '+str(round(self.like_hist[i] - self.like_hist[i-np.min([window_size,i])], 3)))
            stdout.flush()

            window = [i + 1 - np.min([window_size, i]), i]
            
            if self.mle and i > window_size:
                if self.like_hist[i] - self.like_hist[i-window_size] < tolerance:
                    break 

            elif i > window_size:
                if improvement_like[window_size].sum() < tolerance and improvement_prior[window].sum() < tolerance:
                    break 

        return self._model


class OrdinalModelComparison(object):

    def __init__(self, data, cols=['verb', 'frame'], symmetrize=False, learning_rate_default=0.002,
                 learning_rate_dict={'learning_rate_jumps' : 0.00001,
                                     'learning_rate_subj_mult' : 0.0001},
                 minibatchsize=0, maxiter=int(1e6), tolerance=0.01, momentum=0.9, annealing=int(1e5)):
        self._data = data
        self._cols = cols

        self.symmetrize = symmetrize
        
        self.ordreg = {}
        self.lls = []

        self._learning_rate_default = learning_rate_default
        self._learning_rate_dict = learning_rate_dict

        self._minibatchsize = minibatchsize if minibatchsize else self._data.ndatapoints
        self._maxiter = maxiter
        self._tolerance = tolerance
        self._momentum = momentum
        self._annealing = annealing

    def fit(self):
        for equi, ase, mse in itertools.product([True,  False], [True, False], [True, False]):
            if (mse or ase):
                print '\n', equi, mse, ase
                ident = str(equi)+str(ase)+str(mse)
                self.ordreg[ident] = OrdinalRegression(data=self._data,
                                                           cols=self._cols,
                                                           symmetrize=self.symmetrize,
                                                           equidistant=equi,
                                                           additive_subj_effects=ase,
                                                           multiplicative_subj_effects=mse)

                sga_ordreg = StochasticGradientAscent(model=self.ordreg[ident],
                                                      learning_rate_default=self._learning_rate_default,
                                                      learning_rate_dict=self._learning_rate_dict,
                                                      mle=True)

                sga_ordreg.fit(minibatchsize=self._minibatchsize,
                               maxiter=self._maxiter,
                               tolerance=self._tolerance,
                               momentum=self._momentum,
                               annealing=self._annealing)

                aic = -2*sga_ordreg.like_hist.max() + 2*(900+3060+1+86+1)

                if ase and mse:
                    aic += 2*(86+1)
                if equi:
                    aic += 2*1
                else:
                    aic += 2*5

                self.lls.append([equi, ase, mse, sga_ordreg.like_hist.max(), self.ordreg[ident].compute_log_likelihood(), aic])

        self.lls = pandas.DataFrame(self.lls, columns=['equidistant', 'additive', 'multiplicative',
                                                       'minimumll', 'currentll', 'aic'])

    def write_ordregs(self, output):
        for ordreg in self.ordreg.values():
            ordreg.write(output)

    def write_lls(self, output):
        self.lls.to_csv(os.path.join(output, 'ordinal_regression_likelihoods'))

    def write(self, output):
        self.write_ordregs(output)
        self.write_lls(output)            

    @property
    def best_ordreg(self):
        min_row = self.lls[self.lls['aic']==self.lls['aic'].min()]

        return self.ordreg[min_row.equidistant][min_row.additive][min_row.multiplicative]
            
if __name__ == '__main__':
    np.random.seed(seed=23342)
    
    # acceptability = Data(fpath=args.data)
    # acceptability_comparison = OrdinalModelComparison(acceptability)
    # acceptability_comparison.fit()

    similarity = Data(fpath='../data/likert/likert.filtered', cols=['verb0', 'verb1'], sep=',')
    similarity_comparison = OrdinalModelComparison(similarity, cols=['verb0', 'verb1'], symmetrize=True,
                                                   learning_rate_default=0.001, annealing=10000)

    similarity_comparison.fit()
    similarity_comparison.write(os.path.join(args.output, 'similarity'))
    
    # pandas.read_csv(, sep=',')
            
    # data = Data(fpath=args.data, return_shared=False)
    
    # log_likelihoods = defaultdict(list)
    # lfa_history = defaultdict(list)
    # evidence = {}
    # best_lfa = {}
    
    # print '\n\rfeatures\tlog-like\tlog-evidence'
    
    # for i in range(1, 16):        
    #     best_ll = -np.inf
            
    #     for _ in range(args.evidencesample):
    #         lfa = OrdinalFactorAnalysis(data=data, num_of_features=i,
    #                                    feature_sparsity=args.featuresparsity,
    #                                    projection_sparsity=args.projectionsparsity)
    #         ll = lfa.compute_log_likelihood()
            
    #         if not np.isinf(ll):
    #             log_likelihoods[i] = np.append(log_likelihoods[i], ll)
    #             lfa_history[i].append(lfa)
    #             evidence[i] = np.nanmean(log_likelihoods[i])
                
    #             stdout.write('\r'+str(i)+'\t\t'+str(round(ll, 2))+'\t'+str(round(evidence[i], 2)))
    #             stdout.flush()

    #         if ll > best_ll:
    #             best_lfa[i] = lfa
    #             best_ll = ll

    #     best_lfa[i] = best_lfa[i]
                
    #     stdout.write('\r'+str(i)+'\t\t'+str(round(best_ll, 2))+'\t'+str(round(evidence[i], 2)))
    #     stdout.flush()
    #     print '\r'
        
    # data.reset_attributes(return_shared=True)
            
    # evidence = pandas.DataFrame.from_dict(evidence, orient='index')
    # evidence.to_csv(os.path.join(args.output, 'evidence'))
    
    # best_num = np.where(evidence[0]==evidence[0].max())[0][0] + 1

    # weighted_sparsity = {i: np.array(zip(log_likelihoods[i], [lfa.feature_sparsity for lfa in lfa_history[i]])) for i in range(1,16)}
    # pandas.DataFrame(weighted_sparsity[best_num]).to_csv(os.path.join(args.output, 'weighted_sparsity'))
    
    # print '\nThere is the best evidence for '+str(best_num)+' features. Now fitting.'
    # print '\nlog-like\tlog-post\t\tlog-prior'

    # model = best_lfa[best_num]
    # model.reinitialize_objective()

    # sga = StochasticGradientAscent(model=model)
    # sga.fit(minibatchsize=args.minibatchsize, maxiter=args.maxiter, tolerance=args.tolerance)

    # model.write(args.output)

    # # data = Data(fpath=args.data)

    # # lfa = {}
        
    # # for i in range(1, 16):
    # #     print i
        
    # #     lfa[i] = OrdinalFactorAnalysis(data=data, num_of_features=i,
    # #                                   feature_sparsity=args.featuresparsity,
    # #                                   projection_sparsity=args.projectionsparsity,
    # #                                   jump_sparsity=args.jumpsparsity)

    # #     model = lfa[i]
    # #     sga = StochasticGradientAscent(model=model)

    # #     sga.fit(minibatchsize=args.minibatchsize, maxiter=args.maxiter, tolerance=args.tolerance)
        
    # #     lfa[i].write(args.output)
