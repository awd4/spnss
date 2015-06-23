import numpy as np
import random

import spnss.train as train
from spnss.nodes import DiscreteRV, CategoricalNode
import spnss.network as network
from spnss.test.test_rdag import myrdag


def mynet():
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    return network.from_rdag(dag), root, pn1, pn2, cn1, cn2, cn3, cn4

def reset_mynet_params(net):
    root = net.graph.root
    root.children[0].children[0].set_masses([.1, .9])
    root.children[0].children[1].set_masses([.3, .7])
    root.children[1].children[0].set_masses([.2, .8])
    root.children[1].children[1].set_masses([.4, .6])
    root.set_weights([.6, .4])

def test_categorical_node_map_with_dirichlet_prior():
    print '\bcmwdp.',
    drv, cn = DiscreteRV(4), CategoricalNode()
    cn.set_rv(drv)
    data = [0, 0, 2, 3, 3, 3]
    train.categorical_node_map_with_dirichlet_prior(cn, data)
    assert np.allclose(cn.masses, [2./6, 0./6, 1./6, 3./6.])
    train.categorical_node_map_with_dirichlet_prior(cn, data, prior_params=[2,2,2,2])
    assert np.allclose(cn.masses, [3./10., 1./10., 2./10., 4./10.])

def test_categorical_mle():
    print '\bcm.',
    drv, cn = DiscreteRV(5), CategoricalNode()
    cn.set_rv(drv)
    data = [0, 0, 2, 3, 3, 3]
    train.categorical_mle(cn, data)
    assert np.allclose(cn.masses, [2./6, 0./6, 1./6, 3./6, 0./6])

def test_initialize_network():
    print '\bin.',
    net, root, pn1, pn2, cn1, cn2, cn3, cn4 = mynet()
    train.initialize_network(net)
    assert np.allclose(sum(root.weights), 1.0)
    assert np.allclose(sum(cn1.masses), 1.0)
    assert np.allclose(sum(cn2.masses), 1.0)
    assert np.allclose(sum(cn3.masses), 1.0)
    assert np.allclose(sum(cn4.masses), 1.0)

def test_backward_max_generative_trainer():
    print '\bbmgt.',
    net, root, pn1, pn2, cn1, cn2, cn3, cn4 = mynet()
    
    bmgt = train.backward_max_generative_trainer(net, em=True, forward_max=random.choice([False,True]), initialize=False)
    bmgt([[0,0]])
    assert np.allclose(cn3.masses, [(.2+1)/2, .8/2])
    assert np.allclose(cn4.masses, [(.4+1)/2, .6/2])
    assert np.allclose(root.weights, [.6/2, (.4+1)/2])
    bmgt([[0,0]])
    assert np.allclose(cn3.masses, [(.2+2)/3, .8/3])
    assert np.allclose(cn4.masses, [(.4+2)/3, .6/3])
    assert np.allclose(root.weights, [.6/3, (.4+2)/3])
    reset_mynet_params(net)

    bmgt = train.backward_max_generative_trainer(net, em=True, forward_max=random.choice([False,True]), initialize=False)
    bmgt([[float('nan'), float('nan')]])
    assert np.allclose(cn1.masses, [.1/2, (.9+1)/2])
    assert np.allclose(cn2.masses, [.3/2, (.7+1)/2])
    assert np.allclose(root.weights, [(.6+1)/2, .4/2])
    reset_mynet_params(net)

    bmgt = train.backward_max_generative_trainer(net, em=True, forward_max=random.choice([False,True]), initialize=False, normalize=False)
    bmgt([[0,0]])
    assert np.allclose(cn3.masses, [.2+1, .8])
    assert np.allclose(cn4.masses, [.4+1, .6])
    assert np.allclose(root.weights, [.6, .4+1])
    reset_mynet_params(net)

    eta = 2*random.random()
    bmgt = train.backward_max_generative_trainer(net, em=False, eta=eta, forward_max=False, initialize=False)
    bmgt([[0,0]])
    assert np.allclose(cn3.masses, [(.2+eta/.2)/(1+eta/.2), .8/(1+eta/.2)])
    assert np.allclose(cn4.masses, [(.4+eta/.4)/(1+eta/.4), .6/(1+eta/.4)])
    assert np.allclose(root.weights, [.6/(1+eta/.4), (.4+eta/.4)/(1+eta/.4)])
    reset_mynet_params(net)

def test_backward_max_discriminative_gd_trainer():
    print '\bgmdgt(NEEDS TESTS).',
    # TODO: write tests


