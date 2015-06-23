import random
import numpy as np

import ss

def calculate_weights(snw, vw):
    # snw - sum node weights
    # vw  - weights after an EM step attempting to maximize the validation likelihood
    sw = np.ones(len(snw))
    sw /= sw.sum()  # completely-smoothed weights
    # We're going to conceptually draw a line from snw to sw.
    # We'll represent the line as a vector a plus a scalar t times a unit-vector n:
    # x = a + t * n
    a = snw
    n = sw - snw
    if (n==0).all():
        return snw
    # We want to find the point p on the line such that the vector n is perpendicular
    # to the vector pvw, where pvw is the vector from p to the point vw. So the
    # dot product of n and pvw will be zero.
    # p = a + t_p * n
    # pvw = vw - p = vw - (a + t_p * n)
    # dot(n, pvw) = dot(n, vw - (a + t_p * n))
    #             = sum_i n_i * (vw_i - (a_i + t_p * n_i))
    #             = sum_i n_i * (vw_i - a_i - t_p * n_i)
    #             = sum_i n_i * vw_i - n_i * a_i - t_p * n_i**2
    #             = (sum_i n_i * vw_i - n_i * a_i) - t_p * (sum_i n_i**2) = 0
    # Therefore:
    # t_p = (sum_i n_i * vw_i - n_i * a_i) / (sum_i n_i**2)
    tp = (n * (vw - a)).sum() / (n**2).sum()
    if tp <= 0.0:
        # If we don't do this then we're basically putting
        # the validation set in with the training set.
        # Even with this check we're probably incorporating
        # the validation set too much into training.
        return snw
    # So now we find the point p:
    p = a + 0.5*tp * n
    # The problem is that this point could be outside the simplex (by having negative components)
    # Clip the point at the edge of the simplex.
    if (p<0.0).any():
        mi = np.argmin(p)
        # a_i + t_p * n_i = p_i < 0
        # if t_p is changed to -a_i / n_i --> p_i = 0
        tp = -a[mi] / n[mi]
        p = a + tp * n
        p = p.clip(0,1)
    # Return the result
    assert (p>=0.0).all()
    if np.abs(p.sum() - 1.0) > 0.05:
        print 'p', p.sum()
        assert False, 'weights do no sum to one.'
    p /= p.sum()
    return p

def smooth_node(n, sm, old_params):
    w = sm + old_params
    w /= w.sum()
    if n.is_categorical():
        n.set_masses(w)
    else:
        n.set_weights(w)


class Smoother:
    def __init__(self, net, vld, ncuts):
        self.net = net
        self.vld = vld
        self.k = ncuts  # number of times to cut the set of SPN nodes in half
        self.pnodes = [n for n in net.pot if n.is_categorical() or n.is_sum()]
        random.shuffle(self.pnodes)
        self.old_params = None

    def store_old_params(self):
        self.old_params = {n:np.array(n.masses) if n.is_categorical() else np.array(n.weights) for n in self.pnodes}

    def vector_smooth(self):
        edges = ss.compute_edges(self.net, self.vld)
        snl   = [n for n in self.net.pot if n.is_sum()]
        cnl   = [n for n in self.net.pot if n.is_categorical()]
        for sn in snl:
            vw = np.bincount(edges[sn], minlength=len(sn.weights)).astype(np.float)
            if vw.sum() == 0:
                continue
            vw /= vw.sum()
            w = calculate_weights( np.array(sn.weights), vw )
            sn.set_weights(w)
        for cn in cnl:
            vm = np.bincount(edges[cn], minlength=len(cn.masses)).astype(np.float)
            if vm.sum() == 0:
                continue
            vm /= vm.sum()
            m = calculate_weights( np.array(cn.masses), vm )
            cn.set_masses(m)

    def smart_smooth(self, verbose=False):
        edges     = ss.compute_edges(self.net, self.vld)
        pn_scores = sorted( [(ss.compute_score_of(pn, edges),pn) for pn in self.net.pot if pn.is_prod()] )
        for s,pn in pn_scores:
            nodes = pn.children
            self.smooth_nodes(nodes, verbose)

    def smooth(self, verbose=False):
        n = len(self.pnodes)
        size = (n+1) / 2**self.k
        assert size >= 1
        for i in range(0, n, size):
            nodes = self.pnodes[i:i+size]
            self.smooth_nodes(nodes, verbose)

    def smooth_nodes(self, nodes, verbose=False):
        net, vld = self.net, self.vld
        self.store_old_params()
        sm = 0.25
        best_sm = 0.0
        best_llh = net.llh(vld)
        prev_llh = float('-inf')
        if verbose:
            print 'sm:', 0.0, '\tvld llh:', best_llh
        while sm > 0.00001:
            for n in nodes:
                smooth_node(n, sm, self.old_params[n])
            llh = net.llh(vld)
            if verbose:
                print 'sm:', sm, '\tvld llh:', llh
            if llh > best_llh:
                best_llh = llh
                best_sm = sm
            if llh < prev_llh:
                break
            prev_llh = llh
            sm /= 2.0
        for n in nodes:
            smooth_node(n, best_sm, self.old_params[n])


