# distutils: language = c++

from numpy import allclose, prod
import random

from util import assert_raises
cimport spnss.mathutil as mu

from spnss.nodes import Node, SumNode, ProdNode, DiscreteRV, ContinuousRV, GaussianNode, CategoricalNode, IndicatorNode
from spnss.nodes cimport DISCRETE_RV_NATURALS, DISCRETE_RV_INTEGERS

def gt2(i):
    return i > 2

def test_Node():
    print '\bN.',
    nodes = [Node(), SumNode(), ProdNode(), DiscreteRV(), ContinuousRV(), GaussianNode(), CategoricalNode(), IndicatorNode(0)]
    assert [n.is_sum()          for n in nodes] == [False, True, False, False, False, False, False, False], 'ORIGINAL'
    assert [n.is_prod()         for n in nodes] == [False, False, True, False, False, False, False, False]
    assert [n.is_rv()           for n in nodes] == [False, False, False, True, True, False, False, False]
    assert [n.is_discrete()     for n in nodes] == [False, False, False, True, False, False, False, False]
    assert [n.is_continuous()   for n in nodes] == [False, False, False, False, True, False, False, False]
    assert [n.is_distribution() for n in nodes] == [False, False, False, False, False, True, True, True]
    assert [n.is_gaussian()     for n in nodes] == [False, False, False, False, False, True, False, False]
    assert [n.is_categorical()  for n in nodes] == [False, False, False, False, False, False, True, False]
    assert [n.is_indicator()    for n in nodes] == [False, False, False, False, False, False, False, True]

def test_DiscreteRV():
    print '\bD.',
    assert_raises(ValueError, DiscreteRV, -3)
    assert_raises(ValueError, DiscreteRV, 0)
    assert_raises(ValueError, DiscreteRV, 1)
    nodes = [DiscreteRV(), DiscreteRV(15), DiscreteRV(DISCRETE_RV_NATURALS), DiscreteRV(DISCRETE_RV_INTEGERS)]
    for n in nodes:
        n.value = mu.NAN; assert not n.is_known()
        n.value = 0;   assert     n.is_known()
    assert [n.value_in_support(mu.NAN) for n in nodes] == [True, True, True, True]
    assert [n.value_in_support(0) for n in nodes] == [True, True, True, True]
    assert [n.value_in_support(1) for n in nodes] == [True, True, True, True]
    assert [n.value_in_support(2) for n in nodes] == [False, True, True, True]
    assert [n.value_in_support(15) for n in nodes] == [False, False, True, True]
    assert [n.value_in_support(-1) for n in nodes] == [False, False, False, True]
    assert [n.value_in_support(1.5) for n in nodes] == [False, False, False, False]

def test_ContinuousRV():
    print '\bC.',
    assert_raises(ValueError, ContinuousRV, [(0,-1)])
    assert_raises(ValueError, ContinuousRV, [(-10,10), (100,90)])
    assert_raises(ValueError, ContinuousRV, [(-10,10), (100,)])
    nodes = [ContinuousRV(), ContinuousRV([(0,mu.INF)]), ContinuousRV([(1,2)]), ContinuousRV([(0,1), (2,3)])]
    for n in nodes:
        n.value = mu.NAN; assert not n.is_known()
        n.value = 0;   assert     n.is_known()
    assert [n.value_in_support(mu.NAN) for n in nodes] == [True, True, True, True]
    assert [n.value_in_support(0) for n in nodes] == [True, True, False, True]
    assert [n.value_in_support(mu.INF) for n in nodes] == [True, True, False, False]
    assert [n.value_in_support(mu.NINF) for n in nodes] == [True, False, False, False]
    assert [n.value_in_support(1.5) for n in nodes] == [True, True, True, False]
    assert [n.value_in_support(-1) for n in nodes] == [True, False, False, False]
    assert [n.value_in_support(2.5) for n in nodes] == [True, True, False, True]

def test_GaussianNode():
    print '\bG.',
    assert_raises(ValueError, GaussianNode, 0.0, -1.0)
    assert_raises(ValueError, GaussianNode, mu.INF, 1.0)
    assert_raises(ValueError, GaussianNode, mu.NAN, 1.0)
    nodes = [GaussianNode(), GaussianNode(1, 0.1), GaussianNode(-1,5)]
    crv = ContinuousRV()
    for n in nodes:
        n.set_rv(crv)
    crv.value = mu.NAN; assert False not in [(n.forward(), n.forward_log()) == (1.0, 0.0) for n in nodes]
    crv.value = 0;   assert nodes[0].forward() == 1.0/mu.SQRT_TWO_PI and nodes[0].forward_log() == mu.log(1.0/mu.SQRT_TWO_PI)
    crv.value = 1;   assert False not in [allclose(mu.log(n.forward()), n.forward_log()) for n in nodes]
    nv = []
    for n in nodes:
        crv.value = n.mean
        nv.append(n.forward())
    assert nv[1] > nv[0] > nv[2]

def test_CategoricalNode():
    print '\bC.',
    cn = CategoricalNode()
    assert_raises(ValueError, cn.set_rv, DiscreteRV(-1))
    assert_raises(ValueError, cn.set_rv, DiscreteRV(-2))
    drvs = [DiscreteRV(), DiscreteRV(10)]
    nodes = [CategoricalNode(), CategoricalNode()]
    for i in range(len(nodes)):
        nodes[i].set_rv(drvs[i])
        assert nodes[i].masses[0] == 1.0 / drvs[i].num_values
        assert allclose(mu.sum(nodes[i].masses), 1.0)
    for n in nodes:
        tmp = range(n.rv.num_values)
        n.set_masses(tmp)
        assert mu.sum(n.masses) == sum(tmp)
        n.normalize()
        assert allclose(n.masses.sum(), 1.0)
    for n in nodes:
        n.rv.value = mu.NAN
        assert allclose(n.forward(), 1.0)
        assert allclose(n.forward_log(), 0.0)
    for n in nodes:
        n.rv.value = 1000
        n.forward()
        assert mu.isnan(n.forward())       # these lines should raise an exception, but I can't get Cython to do that
        assert mu.isnan(n.forward_log())

def test_IndicatorNode():
    print '\bI.',
    assert_raises(ValueError, IndicatorNode, mu.NINF)
    assert_raises(ValueError, IndicatorNode, mu.INF)
    assert_raises(ValueError, IndicatorNode, mu.NAN)
    nodes = [IndicatorNode(v) for v in [0, 0.5, 3]]
    crv = ContinuousRV()
    drv = DiscreteRV(4)
    nodes[0].set_rv(drv)
    nodes[1].set_rv(crv)
    nodes[2].set_rv(drv)
    crv.value = mu.NAN; drv.value = mu.NAN
    for n in nodes:
        assert n.forward() == 1.0 and n.forward_log() == 0.0
    crv.value = 0.5; drv.value = 0
    for n in nodes[:2]:
        assert n.forward() == 1.0 and n.forward_log() == 0.0
    assert nodes[2].forward() == 0.0 and nodes[2].forward_log() == mu.NINF

def test_SumNode():
    print '\bS.',
    ch = [ProdNode(), SumNode(), GaussianNode()]
    sn = SumNode()
    assert_raises(ValueError, sn.add_child, ContinuousRV())
    assert_raises(ValueError, sn.add_child, DiscreteRV())
    for c in ch:
        sn.add_child(c)
        assert_raises(ValueError, sn.add_child, c)
    sn.clear_children()
    assert len(sn.children) == 0
    for c in ch:
        assert_raises(ValueError, sn.remove_child, c)
    assert_raises(ValueError, sn.set_weights, range(3))
    for c in ch:
        sn.add_child(c)
    sn.normalize()
    assert allclose(sn.weights, [1./3.]*3)
    sn.set_weights(range(3))
    assert mu.sum(sn.weights) == 3
    sn.normalize()
    assert allclose(mu.sum(sn.weights), 1.0)
    for c in ch:
        c.value = 1.0
    assert allclose(sn.forward(), 1.0)
    for c in ch:
        c.value = 0.0
    assert allclose(sn.forward_log(), 0.0)
    for c in ch:
        c.value = random.random()
    sn.set_weights([1]*3)
    assert allclose(sn.forward(), sum([c.value for c in ch]))

    assert sn.forward_max()     == max([sn.weights[i]         * ch[i].value for i in range(len(ch))])
    assert sn.forward_max_log() == max([mu.log(sn.weights[i]) + ch[i].value for i in range(len(ch))])
    for i,c in enumerate(ch):
        c.value = i
    assert sn.max_child() == ch[-1]
    sn.set_weights(range(len(ch))[::-1])
    assert sn.max_weighted_child(random.choice([True, False])) in ch[:2]


def test_ProdNode():
    print '\bP.',
    ch = [ProdNode(), SumNode(), CategoricalNode()]
    pn = ProdNode()
    assert_raises(ValueError, pn.add_child, ContinuousRV())
    assert_raises(ValueError, pn.add_child, DiscreteRV())
    for c in ch:
        pn.add_child(c)
        assert_raises(ValueError, pn.add_child, c)
    pn.clear_children()
    assert len(pn.children) == 0
    for c in ch:
        assert_raises(ValueError, pn.remove_child, c)
    for c in ch:
        pn.add_child(c)
        c.value = 1.0
    assert pn.forward() == 1.0
    for c in ch:
        c.value = 0.0
    assert pn.forward_log() == 0.0
    for c in ch:
        c.value = random.random()
    assert pn.forward() == prod([c.value for c in ch])


def test_copy():
    print '\bc.',
    sn = SumNode()
    pns = [ProdNode(), ProdNode()]
    for pn in pns:
        sn.add_child(pn)
    snc = sn.copy()
    assert id(sn.weights) != id(snc.weights)
    assert allclose(sn.weights, snc.weights)
    assert id(sn.children) != id(snc.children)
    assert set(sn.children) == set(snc.children)


def test_remove_children():
    print '\brc.',
    sn = SumNode()
    sn.add_children([ProdNode() for i in range(3)])
    sn.remove_children(sn.children)
    assert len(sn.children) == 0

    pn = ProdNode()
    pn.add_children([SumNode() for i in range(3)])
    pn.remove_children(pn.children)
    assert len(pn.children) == 0


