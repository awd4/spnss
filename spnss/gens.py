import numpy as np
import logging

import cc
import deps
import nodes
import vv
import train
import knobs
import nbc
import network


class DataSlice(object):
    def __init__(self, data, vvec, vil, is_slice=True):
        assert type(vil) is list
        self.data = data
        self.vvec = vvec
        self.dslice = self.data[:,vil]
        self.vil = vil  # variable index list
        self.is_slice = is_slice
    def cols(self, vil):
        return DataSlice(self.data, self.vvec, vil)
    def schema(self):
        return [self.vvec[vi].num_values for vi in self.vil]
    @property
    def nvars(self):
        return len(self.vil)
    @property
    def ninsts(self):
        return len(self.data.shape[0])


def learn_struct(ds, t):

    if ds.nvars == 1:
        vi = ds.vil[0]
        v = ds.vvec[vi]
        dn = nodes.CategoricalNode() if v.is_discrete() else GaussianNode()
        dn.set_rv(v)
        if v.is_discrete():
            train.categorical_node_map_with_dirichlet_prior(dn, ds.dslice, 2*np.ones(dn.masses.size()))
        else:
            raise Exception, 'cannot handle continuous data at the moment.'
        return dn

    if ds.dslice.shape[0] < knobs.min_instances and ds.nvars > 1:
        pn = nodes.ProdNode()
        for vi in ds.vil:
            dsi = ds.cols([vi])
            dn = learn_struct(dsi, t)
            pn.add_child(dn)
        return pn

    logging.info( 'current slice shape: %s' % str(ds.dslice.shape) )

    dt = deps.dep_tester(ds.dslice, ds.schema(), t, {})
    cl = cc.components(dt, range(ds.nvars))
    if len(cl) == 1:
        assert len(cl[0]) == ds.nvars

    if len(cl) == 1: # or not ds.is_slice:    # cluster
        qa, nc = nbc.kcluster(ds.dslice, ds.schema())
#        qa, nc = nbc.inc_hard_em(ds.dslice, ds.schema())
#        qa, nc = nbc.kcluster_fixed(ds.dslice, 600)
        sn = nodes.SumNode()
        for i in range(nc):
            di = ds.data[qa==i,:]
            dsi = DataSlice(di, ds.vvec, ds.vil)
            child = learn_struct(dsi, t)
            sn.add_child(child)
        sn.set_weights( [float((qa==i).sum()) / len(qa) for i in range(nc)] )
        return sn
    else:
        pn = nodes.ProdNode()
        for c in cl:
            dsc = ds.cols([ds.vil[ci] for ci in c])
            child = learn_struct(dsc, t)
            pn.add_child(child)
        return pn


def learn_spn(data, schema, t):
    vvec = vv.vvec(np.array(schema))
    ds = DataSlice(data, vvec, range(data.shape[1]), False)
    root = learn_struct(ds, t)
    n = network.from_variables_and_root(vvec, root)
    return n


