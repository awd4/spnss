''' Generative structure search (SearchSPN). '''

import numpy as np
import logging

from spnss.mathutil import argmin_fair
from fast import compute_weights_of
from history import NetHistory
import operators as ops
import nbc
import knobs
import cc
import deps


def compute_edges(net, data):
    nodes = net.pot + net.variables
    edges = {n:[] for n in nodes}
    for i,d in enumerate(data):
        net.set(d)
        net.forward_log()
        max_edges = net.backward_max_edges_v2_log()
        for n, ci in max_edges.items():
            edges[n].append( ci )
    for n in edges.keys():
        edges[n] = np.array(edges[n], dtype=np.int)
    return edges


def compute_score_of(pn, edges):
    w = compute_weights_of(pn, edges)
    return np.log(w).sum()


def pn_select(scores, blacklist):
    pnl = [pn for pn in scores.iterkeys() if pn not in blacklist]
    if len(pnl) == 0:
        return None # no suitable product node found
    pnsl = [scores[pn] for pn in pnl]
    pni = argmin_fair(pnsl)
    return pnl[pni]


def chll_select(pn, edges, nmi_cache={}, threshold=0.1):
    ctest = deps.dep_tester2(pn, edges, threshold, nmi_cache)
    ccl = cc.components(ctest, range(len(pn.children)))
    chll = [[pn.children[i] for i in c] for c in ccl]
    chll = [chl for chl in chll if len(chl) >= 2]   # ignore singleton components
    return chll


def is_pn_ok(pn, edges):
    if len(edges[pn]) < knobs.min_instances:
        logging.debug(' product node has too few instances.')
        return False
    if len(pn.children) <= 1:
        logging.debug(' product node has too few children.')
        return False
    return True


def is_chll_ok(pn, chll):
    if len(chll) == 0 or len(chll) == len(pn.children):
        logging.debug(' product node has only singleton child subsets.')
        return False
    return True




class SSData:
    
    def __init__(self, net, data):
        self.edges = compute_edges(net, data)
        self.nmi_cache = {}
        self.pn_scores = {pn:compute_score_of(pn, self.edges) for pn in net.pot if pn.is_prod()}
        self.pn_blacklist = set()

    def update_scores_of(self, pnl):
        for pn in pnl:
            self.pn_scores[pn] = compute_score_of(pn, self.edges)




def select_step(ssdata, threshold):
    while True:
        while True:
            if len(ssdata.pn_blacklist) == len(ssdata.pn_scores):   # give up if all product nodes have been blacklisted
                return None, None

            pn = pn_select(ssdata.pn_scores, ssdata.pn_blacklist)
            if is_pn_ok(pn, ssdata.edges):
                break
            ssdata.pn_blacklist.add(pn)

        chll = chll_select(pn, ssdata.edges, ssdata.nmi_cache, threshold)
        if is_chll_ok(pn, chll):
            break
        ssdata.pn_blacklist.add(pn)

    logging.info( '\tinst x ch: %d x %d' % (len(ssdata.edges[pn]), len(pn.children)) + \
                  '\tchll/total: %s / %d' % (str([len(chl) for chl in chll if len(chl) != 1]), len(pn.children)) )

    return pn, chll



# SS stands for Structure Search
class SS:

    def __init__(self, net, trn, vld, threshold):
        assert trn.ndim == 2
        self.net         = net      # IMPORTANT: we assume that 'net' has already been trained using 'trn'
        self.ssdata      = SSData(net, trn)

        self.net_history = NetHistory(net, vld)

        self.thresh      = threshold

    def print_stats(self, i=None, name=''):
        if i is not None:
            logging.warning('\t================ %d =============== %s' % (i, name))
        msg = '\tsize: %d' % len(self.net.pot)
        msg += '\tvld: %f' % self.net_history.vld_hist[-1]
        msg += '\tthresh: %f' % self.thresh
        logging.warning(msg)

    def step(self):
        ''' Take a step in the search space of SPN graphs. '''

        pn, chll = select_step(self.ssdata, self.thresh)
        if (pn, chll) == (None, None):
            return False
        mol = [ops.MixOp(self.net, pn, chl) for chl in chll]

        for mo in mol:
            # cluster
            data = np.hstack( self.ssdata.edges[c][:,np.newaxis] for c in mo.chl ); assert len(data) >= knobs.min_instances
            nvals = [len(c.weights) if c.is_sum() else len(c.masses) for c in mo.chl]
            qa, nc = nbc.kcluster(data, nvals)
            #qa, nc = nbc.inc_hard_em(data, nvals)

            # change the graph
            mo.connect(qa.max()+1)
            ops.compute_params(mo, qa, self.ssdata.edges, knobs.laplace_smooth)
            ops.adjust_edges(mo, qa, self.ssdata.edges)
            self.ssdata.update_scores_of( [n for n in mo.prod_nodes()] )

        self.net_history.add_op_node_list(mol)

        logging.info('\tncl: %s' % str([len(mo.pnl) for mo in mol]))

        return True

    def step_ahead(self, num_steps):
        for j in xrange(num_steps):
            if self.step() == False:
                logging.warning('\tran out of steps to take.')
                j -= 1
                break
        self.net_history.save_vld()
        return j+1

    def skip_search(self, num_steps):
        ''' This is the main structure search algorithm. '''
        assert num_steps >= 1

        nh = self.net_history

        i0 = 0
        i1 = i0 + self.step_ahead(num_steps)
        i2 = i1 + self.step_ahead(num_steps)
        if nh.vh[i1] <= nh.vh[i0]:
            bni = nh.best_net_index(i0, i1)
            nh.move_to(bni)
            assert self.net == nh.net
            return nh.net

        MAX_STEPS = 100000000
        for i in xrange(MAX_STEPS):
            self.print_stats(i)
            logging.info('\t%f' % nh.vh[-1])
            if nh.vh[i2] <= nh.vh[i1]:
                break
            taken = self.step_ahead(num_steps)
            nh.vh[i0] = None
            i0 = i1
            i1 = i2
            i2 += taken
        if i == MAX_STEPS-1:
            raise Exception, 'skip_search() just took a HUGE number of steps; that is not expected.'

        bni = nh.best_net_index(i0, i2)
        nh.move_to(bni)
        assert self.net == nh.net
        return nh.net


# TODO: DONE! some kind of history with undo/redo capabilities 
# TODO: MOSTLY DONE investigate using Naive-Bayes EM clustering instead of k-means
# TODO: MOSTLY DONE (doesn't seem to work as well). investigate using hamming clustering instead of k-means
# TODO: MOSTLY DONE investigate early-stopping or other criteria to end learning
# TODO: MOSTLY DONE (helps a bit; slows it down) investigate re-computing edges/weights every once in a while


def search_spn(net, trn, vld, t, skip):
    salg = SS(net, trn, vld, t)
    net = salg.skip_search(skip)
    return net


