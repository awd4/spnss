import numpy as np

from nodes import Node, SumNode, ProdNode, CategoricalNode


def connect(pn, chl, sn, pnl, chll, n_sn_ch):
    for i in range(n_sn_ch):    # number of sum node children
        pnl.append( ProdNode() )
        chll.append( [n.copy() for n in chl] )
    for i, n in enumerate(pnl):
        n.add_children( chll[i] )
    sn.add_children(pnl)
    pn.remove_children( chl )
    pn.add_child( sn )


def compute_params(op, qa, edges, smooth=0.0):
    chl, sn, pnl, chll = op.chl, op.sn, op.pnl, op.chll
    assert type(smooth) is float
    assert len(qa) > 0
    for i, pn in enumerate(pnl):
        sn.weights[i] = float((qa==i).sum()) / len(qa)
        for j, c in enumerate(chll[i]):
            e = edges[chl[j]]
            e = e[qa==i]
            if c.is_sum():
                params = np.bincount(e, minlength=len(c.weights)).astype(np.float)
                params /= params.sum()
                assert params.sum() > 0
                c.set_weights(params)
            elif c.is_categorical():
                params = np.bincount(e, minlength=len(c.masses)).astype(np.float) + smooth
                params /= params.sum() + smooth
                assert params.sum() > 0
                c.set_masses(params)


def adjust_edges(op, qa, edges):
    # This updates 'edges', but not exactly as it should; therefore
    # compute_vae should be called every once in a while to
    # re-calibrate 'edges'. Or do the edges get out of whack? Maybe
    # they don't... That would be awesome.
    # Empirical results indicate that 'edges' DOES get out of whack.
    chl, sn, pnl, chll = op.chl, op.sn, op.pnl, op.chll
    edges[sn] = qa.astype(np.int)
    for i, pn in enumerate(pnl):
        edges[pn] = -1*np.ones((qa==i).sum(), dtype=np.int)
    old_edges = [edges[c].copy() for c in chl]
    for i, chl in enumerate(chll):
        for j, c in enumerate(chl):
            edges[c] = old_edges[j][qa==i]




class MixOp:

    def __init__(self, net, pn, chl):
        assert len(set(chl)) > 1
        assert set(chl) <= set(pn.children)
        self.net = net
        self.pn = pn
        self.chl = chl      # original child subset list
        self.sn = SumNode()
        self.pnl = []       # list of product nodes
        self.chll = []      # list of lists of children

    def prod_nodes(self):
        assert self.sn in self.pn.children
        return [self.pn] + self.pnl

    def connect(self, num_sn_ch):
        assert len(self.pnl) == len(self.chll) == 0
        connect(self.pn, self.chl, self.sn, self.pnl, self.chll, num_sn_ch)
        self.net.pot = None

    def undo(self):
        assert len(self.pnl) == len(self.chll) != 0
        self.pn.remove_child( self.sn )
        self.pn.add_children( self.chl )
        self.net.pot = None

    def redo(self):
        assert len(self.pnl) == len(self.chll) != 0
        self.pn.remove_children( self.chl )
        self.pn.add_child( self.sn )
        self.net.pot = None





