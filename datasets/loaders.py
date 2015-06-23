import os
import numpy as np

import creators
from utils import schema_split_helper


BASE_DIR = os.path.dirname( os.path.realpath(__file__) )


''' Load datasets from disk. '''


def load_gd(name, schema=False, split=False):
    GDIR = os.path.join(BASE_DIR, 'gens.domingos/')
    trn = np.loadtxt(GDIR+name+'.ts.data', delimiter=',')
    vld = np.loadtxt(GDIR+name+'.valid.data', delimiter=',')
    tst = np.loadtxt(GDIR+name+'.test.data', delimiter=',')
    assert trn.shape[1] == vld.shape[1] == tst.shape[1]
    assert trn.ndim == vld.ndim == tst.ndim == 2
    n1, n2, n3 = len(trn), len(vld), len(tst)
    n = n1 + n2 + n3
    m = trn.shape[1]
    data = np.zeros((n,m))
    data[:n1,:] = trn
    data[n1:n1+n2,:] = vld
    data[n1+n2:,:] = tst
    return schema_split_helper(data, schema, split, n1, n2, [2]*m)


def load_mnperm(n, schema=False, split=False):
    assert 2 <= n <= 12
    filename = os.path.join(BASE_DIR, 'permanent/mnperm.%.2d.npy'%n)
    try:
        data = np.load(filename)
    except IOError, e:
        data = creators.create_mnperm(n)
        np.save(filename, data)
    m1 = max(1, int(round(data.shape[0]/1.3)))
    m2 = max(1, int(round(m1 * 0.1)))
    return schema_split_helper(data, schema, split, m1, m2, [n]*n)


def gd_function_creator(name):
    def gd_data(schema=False, split=False):
        return load_gd(name, schema, split)
    return gd_data

def mnperm_function_creator(n):
    def mnperm_data(schema=False, split=False):
        return load_mnperm(n, schema, split)
    return mnperm_data


