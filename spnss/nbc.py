''' Naive Bayes Clustering '''

import numpy as np
import random


from spnss.mathutil import argmax_fair, argmin_fair, logsumexp, log
import knobs
from fast import nbmix_likelihood_from_model as fast_nbmix_likelihood_from_model
from history import best_index


def nbmix_model(data, nvals, qa, smooth=0.1):
    data = data.astype(np.int, copy=False)
    nc = qa.max() + 1
    n = data.shape[1]
    m = data.shape[0]
    # compute params for NB models
    lprobs = float('-inf')*np.ones( (n, nc, max(nvals)) )
    priors = np.zeros(nc)
    for i in range(nc):
        di = data[qa==i]
        di_size = float(len(di))
        priors[i] = log(di_size / m)
        for j in range(n):
            bc = np.bincount(di[:,j], minlength=nvals[j])
            for k in range(nvals[j]):
                c = bc[k]
                if c == 0:
                    continue
                lprobs[j,i,k] = log((c + smooth) / (di_size + smooth*nvals[j]))
    return lprobs, priors


def nbmix_likelihood_from_model(data, lprobs, priors):
    nc = lprobs[0].shape[0]
    lp = np.zeros(nc)
    ll = 0.0
    for d in data:
        lp[:] = 0.0
        for j in range(data.shape[1]):
            lp += lprobs[j,:,d[j]]
        lp += priors
        ll += logsumexp(lp)
    return ll


def nbmix_likelihood(data, nvals, qa, smooth=0.1):
    lprobs, priors = nbmix_model(data, nvals, qa, smooth)
    return fast_nbmix_likelihood_from_model(np.asarray(data, order='c'), lprobs, priors)
#    nbll = nbmix_likelihood_from_model(data, lprobs, priors)
#    assert np.allclose( nbll, fast_nbmix_likelihood_from_model(data, lprobs, priors) )
#    exit()



def max_likelihood_qa(data, nvals, qa, approximate=False, smooth=0.1):
    nc = qa.max() + 1
    sil = np.zeros(nc)
    for i in range(nc):
        di = data[qa==i].astype(np.int)
        total = float(len(di))
        si = 0.0
        for j in range(di.shape[1]):
            bc = np.bincount(di[:,j])
            for c in bc:
                if c == 0:
                    continue
                si += c*log(c / total)
        si += log(total / len(data)) # cluster prior
        sil[i] = si
    s = sil.sum() if approximate else logsumexp(sil)
    return s 


def cluster_and_score(data, nvals, alg, k):
    qa = alg(data, k)
    nc = qa.max()+1
    if nc < 2:
        return None, None
    #s = max_likelihood_qa(data, nvals, qa, False)
    s = nbmix_likelihood(data, nvals, qa, 0.1)
    s -= knobs.cluster_penalty * nc * data.shape[1] # why data.shape[1]?
    return qa, s
 

def cluster_with_automatic_k_selection_linear(data, nvals, alg):
    best_score, best_qa, best_k = float('-inf'), None, 2
    for k in xrange( 2, max(3, len(data)/2) ):
        qa, s = cluster_and_score(data, nvals, alg, k)
        if qa is None:
            break
        if s > best_score:
            best_score = s
            best_qa = qa
            best_k = k
        elif k > 6 and k > 1.5 * best_k:
            break
        if best_score == 0.0:
            break
    if best_qa is None:
        data = np.array(data)
        raise Exception, 'cluster_with_automatic_k_selection_linear(): was not able to cluster into two or more clusters'
    return best_qa, best_qa.max()+1


def cluster_with_automatic_k_selection_skip(data, nvals, alg, skip):
    assert skip > 1

    def insert_score(k):
        if sqah[k] is None:
            sqah[k] = cluster_and_score(data, nvals, alg, k)[::-1]

    sqah = [None] * (4 + 2*skip)
    i, m, j = 2, 2+skip, 2+2*skip
    insert_score(i)
    insert_score(m)
    insert_score(j)
    si, sm, sj = [sqah[k][0] for k in [i,m,j]]

    if sm <= si:
        ib = best_index(i, m, lambda k: sqah[k][0] if sqah[k] is not None else None, insert_score, float('-inf'))
        sb, qab = sqah[ib]
        return qab, qab.max()+1

    while j < len(data)/2:
        if sj <= sm:
            break
        sqah.extend( [None]*skip )
        i = m
        m = j
        j += skip
        insert_score(j)
        si, sm, sj = [sqah[k][0] for k in [i,m,j]]

    ib = best_index(i, j, lambda k: sqah[k][0] if sqah[k] is not None else None, insert_score, float('-inf'))
    sb, qab = sqah[ib]
    return qab, qab.max()+1


def cluster_with_automatic_k_selection_binary(data, nvals, alg):
    assert len(data) >= 8
    best_score, best_qa, best_k = float('-inf'), None, 2
    intervals = [(2, min(2000, len(data) / 2))]  # assume no cluster should have fewer than two data instances; limit to 2000 clusters or fewer
    sqah = [None for i in range(3 + len(data)/2)]
    while len(intervals) > 0:
        i, j = intervals.pop()
        m = int((i + j) / 2) 
        for k in [i, m, j]:
            if sqah[k] is None:
                sqah[k] = cluster_and_score(data, nvals, alg, k)[::-1]
        si, sm, sj = [sqah[k][0] for k in [i,m,j]]
        ib = [i,m,j][np.argmax( [sqah[k][0] for k in [i,m,j]] )]
        sb, qab = sqah[ib]

        if sb > best_score:
            best_score = sb
            best_qa = qab
            best_k = ib
        elif sb == best_score and ib < best_k:
            best_qa = qab
            best_k = ib

        if sm > max(si, sj):
            if i + 1 < m: intervals.append( (i, m) )
            if m + 1 < j: intervals.append( (m, j) )
        elif sm == si and sm >= sj:
            continue
        elif sm == sj and sm > si:
            if i + 1 < m: intervals.append( (i, m) )
        elif best_score > max(si, sj):
            continue
        elif si >= sj:
            if i + 1 < m: intervals.append( (i, m) )
        else:
            if m + 1 < j: intervals.append( (m, j) )
    if best_qa is None:
        raise Exception, 'cluster_with_automatic_k_selection_binary(): was not able to cluster into two or more clusters'
    return best_qa, best_qa.max()+1


def cluster_with_automatic_k_selection(data, nvals, alg):
    #return cluster_with_automatic_k_selection_linear(data, nvals, alg)
    if len(data) <= 30:
        return cluster_with_automatic_k_selection_linear(data, nvals, alg)
    else:
        #return cluster_with_automatic_k_selection_binary(data, nvals, alg)
        return cluster_with_automatic_k_selection_skip(data, nvals, alg, 20)


def kcluster(data, nvals):
    data = data.astype(np.float)
    from vlfwrap import KMeans
    km = KMeans()
    km.repetitions = 1
    def kmeans(data, k):
        km.cluster(data, k)
        return km.quantize(data)
    return cluster_with_automatic_k_selection(data, nvals, kmeans)

def kcluster_fixed(data, k):
    data = data.astype(np.float)
    from vlfwrap import KMeans
    km = KMeans()
    km.repetitions = 1
    km.cluster(data, k)
    qa = km.quantize(data)
    nc = qa.max() + 1
#    if nc != k and len(data) >= k:
#        qa = np.arange(len(data))
#        qa %= k
#        nc = k
    return qa, nc 

def hamming(x1, x2):
    assert len(x1) == len(x2)
    return (x1 != x2).sum()

def ham_cluster(data, k):
    m, n = data.shape
    assert 1 <= k <= m
    pts = range(m)
    p = random.choice(pts)
    pts.remove(p)
    centers = [p]
    distances = [[hamming(data[i], data[p])] for i in range(m)]
    for i in range(k-1):
        pi = argmax_fair([min(distances[q]) for q in pts])
        p = pts[pi]
        for q in pts:
            distances[q].append( hamming(data[q], data[p]) )
        centers.append(p)
        pts.remove(p)
    a = [argmin_fair(distances[q]) for q in range(m)]  # assignments
    return np.array(a)

def hcluster(data, nvals):
    data = data.astype(np.float)
    return cluster_with_automatic_k_selection(data, nvals, ham_cluster)









from fast import fast_ll
class Cluster():
    def __init__(self, nvals):
        self.nvals = nvals
        self.smoo = 0.1
        self.sstats = {}
        self.size = 0

    def add_inst(self, inst):
        for i in xrange(len(self.nvals)):
            v = inst[i]
            if i not in self.sstats:
                self.sstats[i] = [0]*self.nvals[i]
            self.sstats[i][v] = self.sstats[i][v] + 1
        self.size += 1

    def remove_inst(self, inst):
        self.size -= 1
        for i in xrange(len(self.nvals)):
            v = inst[i]
            self.sstats[i][v] = self.sstats[i][v] - 1

    def ll(self, inst, best_cll=float('-inf')):
        sstats = self.sstats
        l = 0.0
        for i in xrange(len(self.nvals)):
            v = inst[i]
            w = sstats[i][v]
            l += log((w + self.smoo) / (self.size + self.nvals[i]*self.smoo))
            if l < best_cll:
                return l
        return l

    def fast_ll(self, inst, best_cll):
        return fast_ll(inst, self.sstats, self.nvals, self.size, self.smoo, best_cll)

    def is_empty(self):
        return self.size == 0


def penalized_ll(nbcs, data):
    ll = 0.0
    cluster_priors = [log(1.0 * c.size / len(data)) for c in nbcs]
    for inst in data:
        vals = [cluster_priors[i] + nbcs[i].fast_ll(inst, float('-inf')) for i in range(len(nbcs))]
        ll += logsumexp( vals )
    ll -= knobs.cluster_penalty * len(nbcs) * data.shape[1]
    return ll



def new_ll(nvals, smooth=0.1):
    return sum( log((1. + smooth) / (1. + v * smooth)) for v in nvals )


def inc_hard_em(data, nvals):
    num_runs = 10
    num_em_its = 4
    best_ll = float('-inf')
    best_qa = np.zeros(len(data), dtype=np.int)

    instance_order = range(len(data))

    new_cluster_ll = new_ll(nvals, 0.1)
    new_cluster_penalized_ll = -knobs.cluster_penalty * 1 * len(nvals) + new_cluster_ll
    for r in range(num_runs):
        #print 'EM Run', r, 'with', len(instance_order), 'insts', len(nvals), 'vars'
        nbcs = []

        random.shuffle(instance_order)

        inst_cluster_map = {}

        it = 0
        while it < num_em_its:
            it += 1
            ll = 0
            for i in instance_order:
                inst = data[i]
                prev_cluster = inst_cluster_map.pop(i, None)
                if prev_cluster is not None:
                    prev_cluster.remove_inst(inst)
                    if prev_cluster.is_empty():
                        nbcs.remove( prev_cluster )
                best_cll = new_cluster_penalized_ll
                best_cluster = None
                for c in nbcs:
                    cll = c.fast_ll(inst, best_cll)
                    if cll > best_cll:
                        best_cll = cll
                        best_cluster = c

                # make new cluster if best_cll is not greater than penalty
                if best_cluster is None:
                    best_cluster = Cluster(nvals)
                    nbcs.append(best_cluster)
                    if len(nbcs) > 2000 and len(data) > 10000:
                        print 'Too many clusters, increase CP penalty'
                        exit()

                best_cluster.add_inst(inst)
                inst_cluster_map[i] = best_cluster

                ll += best_cll

            if len(nbcs) == 1:
                it = 0
                new_cluster_penalized_ll *= 0.5
            # end em
        ll = penalized_ll(nbcs, data)        
        if ll > best_ll:
            best_ll = ll
            c_indices = {c:i for i,c in enumerate(nbcs)}
            best_qa[:] = [c_indices[inst_cluster_map[i]] for i in range(len(data))]
    return best_qa, best_qa.max()+1



