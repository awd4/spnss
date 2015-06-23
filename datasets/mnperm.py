# Markov Network that approximates the "matrix permanent" distribution
#
# The Markov Network (MN) in this file is a pair-wise MN on
# a complete graph (every pair of vertices is adjacent).
#
# The potentials are as follows. Say we have variables X and Y
# and each can take 3 values. Then we set up the potential phi(X,Y)
# so that X wants to take any value other than the value Y takes
# and vice versa. We put some constant (say 10) on the disagreeing
# values and 1 on the agreeing values. So the potential would be:
#
# X  Y  phi(X,Y)
# 0  0  1
# 0  1  10
# 0  2  10
# 1  0  10
# 1  1  1
# 1  2  10
# 2  0  10
# 2  1  10
# 2  2  1
# 
# Now we'll use Gibbs sampling to sample from this network. The
# unnormalized joint is:
#
# Q = prod_i phi_i(D_i)
#
# where D_i is a pair of variables and i ranges over all the pairs.
# The complete conditional P(X|Y=y), where X is a single variable and
# Y is the set of all other variables, is:
#
# P(X|Y=y) = Q(X,y) / sum_X Q(X,y)
#          = prod_i phi_i(D_i~y) / sum_X prod_i phi_i(D_i~y)
#
# where D_i~y is a setting of the variables D_i consistent with Y=y.
# Further,
#
# P(X|Y=y) = ph_1(D_1~y) * ... * phi_k(D_k~y) / sum_X ( phi_1(D_1~y) * ... * phi_k(D_k~y) )
#
#            prod_{X not in D_i} phi_i(D_i~y) * prod_{X in D_i} ph_i(D_i~y) 
#          = ------------------------------------------------------------------------
#            prod_{X not in D_i} phi_i(D_i~y) * sum_X ( prod_{X in D_i} ph_i(D_i~y) )
#
#                prod_{X in D_i} ph_i(D_i~y) 
#          = -------------------------------------
#            sum_X ( prod_{X in D_i} ph_i(D_i~y) )
#

import itertools
import random
import numpy as np


def pval(potentials, i, j, vi, vj):
    assert i != j
    if i > j:
        i, j = j, i
        vi, vj = vj, vi
    return potentials[(i,j)][vi, vj]

def uprob(potentials, n, z):
    variables = range(n)
    up = 1.0
    for i in range(n):
        for j in range(i+1,n):
            p = pval(potentials, i, j, z[i], z[j])
            up *= p
    return up

def Z(potentials, n):
    partition = 0.0
    for z in itertools.product(range(n), repeat=n):
        up = uprob(potentials, n, z)
        partition += up
    return partition

#                prod_{X in D_i} ph_i(D_i~y) 
#          = -------------------------------------
#            sum_X ( prod_{X in D_i} ph_i(D_i~y) )
def complete_conditional(X, y, potentials, n, nvals):
    # X in {0,...,n-1}
    # y = {0:val_0, ..., X-1:val_{X-1}, X+1:val_{X+1}, ..., n-1:val_{n-1}}
    if type(y) is list:
        assert len(y) == n-1
        y.insert(X, -1)
        y = {i:y[i] for i in range(len(y))}
        del y[X]
    ccd = [0.0]*nvals    # complete conditional distribution
    b = 0.0
    for v in range(nvals):
        t = 1.0
        for i in range(n):
            if i == X: continue
            t *= pval(potentials, X, i, v, y[i])
        b += t
        ccd[v] = t
    ccd = [p/b for p in ccd]
    return ccd

def gibbs_sample(m, b, potentials, n, vals):
    # m - number of samples to take
    # b - the burn-in number of samples
    variables = range(n)
    y = {i:random.choice(vals) for i in variables}
    X = random.choice(variables)
    samples = [[y[j] for j in variables]]
    del y[X]
    for i in range(m+b-1):
        cc = complete_conditional(X, y, potentials, n, len(vals))
        v = np.random.choice( vals, p=cc )
        y[X] = v
        s = [y[j] for j in variables]
        samples.append(s)
        X = random.choice(variables)
        del y[X]
    samples = samples[b:]
    random.shuffle(samples)
    return samples

def perm_potentials(n, C):
    potentials = {}
    for i in range(n):
        for j in range(i+1,n):
            p = 100*np.ones((n, n))
            p -= 99*np.eye(n)
            potentials[(i,j)] = p
    return potentials


if __name__ == '__main__':
    n = 4
    vals = range(n)
    nvals = len(vals)

    potentials = perm_potentials(n, 10)

    gs = gibbs_sample(600, 2, potentials, n, vals)

    import collections
    c = collections.Counter([tuple(s) for s in gs])
    d = sorted(c.items())
    for k,v in d:
        print k, v

    #y = [0,2,1]
    #X = 1
    #print complete_conditional(X, y, potentials, n, nvals)
    #exit()


