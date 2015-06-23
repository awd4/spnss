import numpy as np

from fast import nmi as fast_nmi


def nmi(d1, d2, n1, n2):
    c = np.bincount(d2 + d1*n2, minlength=n1*n2).reshape((n1,n2))
    a = c.sum(axis=1)
    b = c.sum(axis=0)
    T = float(a.sum())
    d = np.dot(a[:,np.newaxis], b[np.newaxis,:]) / (T * T)
    p = c / T
    pnz = p[p!=0]
    plp = pnz * np.log(pnz)
    m = (plp - pnz * np.log(d[p!=0])).sum()
    e = -plp.sum()
#    if (e1 == e1).all() and (e1 == e1[0]).all():
#        assert m == 0 and e == 0
    if m == 0 and e == 0: # take care of case where (ei==ej).all() and (ei == ei[0]).all()
        n = -1.0
    else:
        n = m / e
    if n > 1.0:
        if n < 1.0 + 1e-10:
            n = 1.0
        else:
            raise Exception, 'nmi value too high: %f' % n
    return n

def max_nmi(data, nvals):
    assert len(nvals) == data.shape[1]
    maxnmi = float('-inf')
    for i in range(len(nvals)):
        for j in range(i+1, len(nvals)):
            n = nmi(data[:,i], data[:,j], nvals[i], nvals[j])
            if n > maxnmi:
                maxnmi = n
    return maxnmi


def dep_tester(data, nvals, threshold, nmi_cache={}):
    def deptest(i, j):
        if   (i,j) in nmi_cache:
            n = nmi_cache[(i,j)]
        elif (j,i) in nmi_cache:
            n = nmi_cache[(j,i)]
        else:
            n, should_cache = fast_nmi(data[:,i], data[:,j], nvals[i], nvals[j])
            nmi_cache[(i,j)] = n
            nmi_cache[(j,i)] = n
        return n > threshold
    return deptest

def dep_tester2(pn, edges, threshold, nmi_cache={}):
    nvals = [len(c.weights) if c.is_sum() else len(c.masses) for c in pn.children]
    def deptest(i, j):
        ci, cj = pn.children[i], pn.children[j]
        if (ci, cj) in nmi_cache:
            n = nmi_cache[(ci, cj)]
        elif (cj, ci) in nmi_cache:
            n = nmi_cache[(cj, ci)]
        else:
            ei, ej = edges[ci], edges[cj]
            n, should_cache = fast_nmi(ei, ej, nvals[i], nvals[j])
#            assert np.allclose( n, nmi(ei, ej, nvals[i], nvals[j]) )
#            exit()
            if should_cache:    # this restrains the size of the cache
                nmi_cache[(ci, cj)] = n
                nmi_cache[(cj, ci)] = n
        return n > threshold
    return deptest


