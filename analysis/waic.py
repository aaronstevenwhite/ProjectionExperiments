import numpy as np

def construct_prob_trace(trace, responses):
    return trace[:,range(responses.shape[0]),responses]

def compute_lppd(prob_trace):
    return np.log(prob_trace.mean(axis=0)).sum()

def compute_p_waic(prob_trace, method=2):
    if method == 1:
        mean_log = np.log(prob_trace).mean(axis=0)
        log_mean = np.log(prob_trace.mean(axis=0)) 
        
        return 2 * (log_mean - mean_log).sum() 
    elif method == 2:
        return np.log(prob_trace).var(axis=0).sum()
    else:
        raise ValueError, 'method parameter must be either 1 or 2'        

def compute_waic(prob_trace, method=2):
    lppd = compute_lppd(prob_trace=prob_trace)
    p_waic = compute_p_waic(prob_trace=prob_trace, method=method)

    return -2 * (lppd - p_waic)
