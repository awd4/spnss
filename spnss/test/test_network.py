# distutils: language = c++

from numpy import prod
from numpy.random import rand
from scipy.stats import norm

from util import assert_raises

import spnss.network as network
from spnss.network import Network
from spnss.nodes import DiscreteRV


def test_basic():
    print '\bb.',
    n = Network()
    assert_raises(AttributeError, n.set, (0,1,2))

    n = network.independent([DiscreteRV(4) for i in range(10)])
    n.set((0,1,2,3,0,1,2,3,0,1))
    assert n.forward() == (1./4)**10, str(n.forward()) + ',' + str((1./4)**10)

    n = network.independent('4cx1')
    n.set((0,1,2,3))
    assert n.forward() == prod([norm.pdf(x) for x in (0,1,2,3)])

    n = network.independent('4d3')
    assert_raises(ValueError, n.set, (0,1,2,3))

def test_scopes():
    print '\bs.',
    net = network.simple()
    scopes = net.scopes()
    for c in net.graph.root.children:
        assert scopes[net.graph.root] == scopes[c] == set([0,1])
    assert False not in [len(scopes[n]) == 1 if n.is_distribution() else len(scopes[n]) == 2 for n in net.pot]

def test_pot():
    print '\bp.',
    net = network.simple()
    assert len(net.pot) == 7
    net.pot = None
    assert len(net.pot) == 7

def test_save_load():
    print '\bsl.',
    net1 = network.simple()
    for n in net1.pot:
        if n.is_sum():
            n.set_weights(rand(len(n.weights)))
        elif n.is_categorical():
            n.set_masses(rand(len(n.masses)))
    network.save(net1, 'tmp.spn')
    net2 = network.load('tmp.spn')
    assert len(net1.pot) == len(net2.pot)
    for i in range(len(net1.pot)):
        assert id(net1.pot[i]) != id(net2.pot[i])
    for d in [(0,0), (0,1), (1,0), (1,1)]:
        assert net1.forward(d) == net2.forward(d)



