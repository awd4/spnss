import argparse
import cPickle as pickle
import time
import numpy as np
import random

import datasets
import spnss.network as network
import spnss.train as train 
import spnss.smooth as smooth 
import spnss.knobs as knobs 


names = {0:'accidents', 1:'ad', 2:'baudio', 3:'bbc', 4:'bnetflix', 5:'book', 6:'c20ng', 7:'cr52', 8:'cwebkb', 9:'dna', 10:'jester', 11:'kdd', 12:'kosarek', 13:'msnbc', 14:'msweb', 15:'nltcs', 16:'plants', 17:'pumsb_star', 18:'tmovie', 19:'tretail', 20:'mnist', 21:'mnist14', 22:'mnist07', 23:'mnist04', 24:'mnist02'}

def load_data(name):
    trn, vld, tst, schema = datasets.intify( *getattr(datasets, name)(split=True, schema=True) )
    #trn = trn.round().astype(np.int)
    #vld = vld.round().astype(np.int)
    #tst = tst.round().astype(np.int)
    return trn, vld, tst, schema

def seed_network(trn, schema):
    net = network.independent(np.array(schema))
    for i,cn in enumerate(net.graph.root.children):
        train.categorical_node_map_with_dirichlet_prior(cn, trn[:,i], 2*np.ones(cn.masses.size()))
    return net

def smooth_network(net, vld, verbose=True):
    sm = smooth.Smoother(net, vld, 0)
    sm.smooth(verbose)


def thresh_float(t):
    t = float(t)
    if t != -1 and (t < 0.0 or t > 1.0):
        raise argparse.ArgumentTypeError('%r not in range (0.0, 1.0)'%t)
    return t

def vq_parser():
    vqp = argparse.ArgumentParser(add_help=False)
    group = vqp.add_mutually_exclusive_group()
    group.add_argument('-v', help='increase verbosity level of the output', action='count', default=1)
    group.add_argument('-q', help='decrease verbosity level of the output', action='count', default=0)
    return vqp

def t_cp_parser():
    tcp = argparse.ArgumentParser(add_help=False)
    tcp.add_argument('-t', '--threshold', type=thresh_float, help='nmi dependence threshold (must be between 0 and 1 inclusive)')
    tcp.add_argument('-cp', '--cluster-penalty', type=float, help='penalty on the number of clusters in the Naive Bayes model')
    return tcp

def io_parser():
    iop = argparse.ArgumentParser(add_help=False)
    iop.add_argument('-i', '--input-file', help='input file for loading an SPN')
    iop.add_argument('-o', '--output-file', help='output file to store the resulting SPN')
    return iop

def dataset_parser():
    dp = argparse.ArgumentParser(add_help=False)
    dp.add_argument('dataset', help='the name of the dataset to use in training, validation, and testing')
    return dp

def get_name(dataset):
    try:
        name = names[int(dataset)]
    except ValueError:
        name = dataset
    return name

def load_net(filename):
    net = None
    if filename is not None:
        print 'loading...'
        try:
            with open(filename, 'rb') as fp:
                spn = pickle.load(fp)
                net = network.from_data(spn)
        except IOError, e:
            print 'WARNING: could not open the input file'
    return net

def save_net(net, filename, prompt=False):
    if filename is not None:
        if prompt:
            ri = raw_input('Keep this SPN? ')
            if ri.lower() not in ['y', 'yes']:
                return
        print 'saving...'
        spn = network.to_data(net)
        with open(filename, 'wb') as fp:
            pickle.dump(spn, fp)


