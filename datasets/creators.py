import random
import numpy as np

import mnperm
from utils import mnperm_trn_size, schema_split_helper

''' Create datasets and save them to disk. '''


def get_state():
    return random.getstate(), np.random.get_state()

def set_state(state):
    rs, ns = state
    random.setstate(rs)
    np.random.set_state(ns)

def seed(rseed=None, nseed=None):
    rseed = 128502187 if rseed is None else rseed
    nseed = 71629476  if nseed is None else nseed
    state = get_state()
    random.seed(rseed)
    np.random.seed(nseed)
    return state


def sample_mnperm(n, m=100, b=100, C=10, skip=1, va_size=0, te_size=0, schema=False, split=False):
    ''' Markov Network version of the uniform "permanent" distribution. '''
    vals = range(n)
    data = np.zeros((m + va_size + te_size, n))
    potentials = mnperm.perm_potentials(n, C)
    gs         = mnperm.gibbs_sample(skip*(m+va_size+te_size), b, potentials, n, vals)
    for i in range(m+va_size+te_size):
        data[i] = gs[i*skip]
    return schema_split_helper(data, schema, split, m, va_size, [n]*n)


def create_mnperm(n):
    old_state = seed() # Ensure that the same dataset is generated each time
    burn_in, C, skip = 1000, 10, 2*n
    m1 = mnperm_trn_size(n)
    m2 = max(1, int(round(m1 * 0.1)))
    m3 = max(1, int(round(m1 * 0.2)))
    data = sample_mnperm(n, m1, burn_in, C, skip, m2, m3, False, False)
    set_state(old_state)
    return data


