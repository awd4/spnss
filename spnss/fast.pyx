# distutils: language = c++
# cython: profile=True

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from spnss.mathutil cimport log, logsumexp_pointer
#from mathutil import argmax_fair, argmin_fair, logsumexp, logaddexp

from spnss.nodes cimport Node, SumNode, ProdNode, CategoricalNode


def forward(list nodes, bint maxspace, bint logspace):
    cdef Node n
    for n in nodes:
        if logspace:
            if maxspace and n.is_sum():
                n.forward_max_log()
            else:
                n.forward_log()
        else:
            if maxspace and n.is_sum():
                n.forward_max()
            else:
                n.forward()
    return nodes[-1].value


def compute_weights_of(ProdNode pn, dict edges):
    c0 = pn.children[0]#; assert [len(edges[c]) for c in pn.children].count(len(edges[c0])) == len(pn.children)
    cdef int j, k, e, nc, ne
    cdef Node c
    cdef ndarray el
    cdef SumNode sn
    cdef CategoricalNode cn
    cdef ndarray[float_t, ndim=2] w
    nc = len(pn.children)
    ne = len(edges[c0])
    w = np.zeros( (ne, nc) )
    for j in range(nc):
        c = pn.children[j]
        el = edges[c]
        for k in range(ne):
            e = el[k]
            if c.is_sum():
                sn = <SumNode>c
                w[k,j] = sn.weights[e]
            else:
                cn = <CategoricalNode>c
                w[k,j] = cn.masses[e]
    return w
#def compute_weights_of_slow(pn, edges):
#    c0 = pn.children[0]#; assert [len(edges[c]) for c in pn.children].count(len(edges[c0])) == len(pn.children)
#    w = np.zeros( (len(edges[c0]), len(pn.children)) )
#    for j, c in enumerate(pn.children):
#        w[:,j] = [c.weights[e] if c.is_sum() else c.masses[e] for e in edges[c]]
#    return w


def fast_ll(np.ndarray inst, dict sstats, list nvals, int size, double smoo, double best_cll):
    cdef double l, w
    cdef int i, nv, v, d
    cdef list counts
    nv = len(nvals)
    l = 0.0
    for i in range(nv):
        d = nvals[i]
        v = inst[i]
        counts = sstats[i]
        w = counts[v]
        l += log((w + smoo) / (size + d*smoo))
        if l < best_cll:
            return l
    return l




def nmi(np.ndarray[np.int_t, ndim=1] d1, np.ndarray[np.int_t, ndim=1] d2, int n1, int n2):
    cdef int *a
    cdef int *b
    cdef int *c
    cdef double *d
    cdef double T, T2, m, e, p, l
    cdef unsigned int i, j
    cdef int ci, di, dj

    c = <int *>PyMem_Malloc(n1*n2 * sizeof(int))
    for i in range(n1*n2): c[i] = 0
    for i in range(d1.shape[0]):
        di = d1[i]
        dj = d2[i]
        c[di*n2 + dj] += 1

    a = <int *>PyMem_Malloc(n1 * sizeof(int))
    b = <int *>PyMem_Malloc(n2 * sizeof(int))
    for i in range(n1): a[i] = 0
    for i in range(n2): b[i] = 0
    T = 0.0
    for i in range(n1):
        for j in range(n2):
            ci = c[i*n2 + j]
            a[i] += ci
            b[j] += ci
        T += a[i]
    T2 = T * T

    d = <double *>PyMem_Malloc(n1*n2 * sizeof(double))
    for i in range(n1):
        for j in range(n2):
            d[i*n2 + j] = (<double>a[i] * b[j]) / T2

    m = 0.0
    e = 0.0
    for i in range(n1):
        for j in range(n2):
            ci = c[i*n2 + j]
            if ci == 0:
                continue
            p = c[i*n2 + j] / T
            pl = p * log(p)
            m += pl - p * log( d[i*n2 + j] )
            e += pl
    e *= -1.0

    PyMem_Free(a)
    PyMem_Free(b)
    PyMem_Free(c)
    PyMem_Free(d)

    if m == 0 and e == 0: # take care of case where (ei==ej).all() and (ei == ei[0]).all()
        n = -1.0
    else:
        n = m / e
    if n > 1.0:
        if n < 1.0 + 1e-10:
            n = 1.0
        else:
            raise Exception, 'nmi value too high: %f' % n
    # return the nmi value and True if it might be worth it to cache the result
    return n, d1.shape[0] > 200 or n1*n2 > 100



def nbmix_likelihood_from_model(np.ndarray[np.float64_t, ndim=2, mode='c'] data not None,
                                np.ndarray[np.float64_t, ndim=3, mode='c'] lprobs not None,
                                np.ndarray[np.float64_t, ndim=1, mode='c'] priors not None):
    cdef unsigned int i, j, k, v, r, nc, num_vars, num_inst, num_clst, num_vals
    cdef double *lp
    cdef double *dp
    cdef double *pp
    cdef double *lpp

    dp = &data[0,0]
    lpp= &lprobs[0,0,0]
    pp = &priors[0]

    num_vars = data.shape[1]
    num_inst = data.shape[0]
    num_clst = lprobs.shape[1]
    num_vals = lprobs.shape[2]
    nc = num_clst

    lp = <double *>PyMem_Malloc(nc * sizeof(double))
    for i in range(nc): lp[i] = 0.0

    ll = 0.0
    for k in range(num_inst):
        
        for i in range(nc): lp[i] = 0.0

        for j in range(num_vars):

            v = <unsigned int>(dp[k*num_vars + j])

            r = j * (num_clst * num_vals)
            for i in range(nc):
                lp[i] += lpp[r + i*num_vals + v]

        for i in range(nc):
            lp[i] += pp[i]

        ll += logsumexp_pointer(lp, nc)

    PyMem_Free(lp)
    return ll


