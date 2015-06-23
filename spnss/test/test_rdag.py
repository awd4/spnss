# distutils: language = c++

#from numpy import prod
#from scipy.stats import norm

from util import assert_raises

import spnss.rdag
from spnss.rdag import RootedDAG
from spnss.nodes import SumNode, ProdNode, DiscreteRV


''' This example graph is inspired from the Wikipedia page "Topological Sorting". The
    difference here is that a root node has been added and the nodes have been labeled
    with "+" or "x".

              +          sn1
            / |  \ 
           x  x   x      pn1, pn2, pn3
           |\/__ /|
           +  _ + |      sn2, sn3
          /|\/____|
         x x      x      pn4, pn5, pn6
'''
def dummydag():
    # WARNING: do not change this code; many unit tests (including ones not in this module) rely on it being just so.
    sn1, sn2, sn3 = SumNode(), SumNode(), SumNode()
    pn1, pn2, pn3, pn4, pn5, pn6 = ProdNode(), ProdNode(), ProdNode(), ProdNode(), ProdNode(), ProdNode()
    sn1.add_children([pn1, pn2, pn3])
    pn1.add_children([sn2, sn3])
    pn2.add_child(sn2)
    pn3.add_children([sn3, pn6])
    sn2.add_children([pn4, pn5, pn6])
    sn3.add_child(pn5)
    return RootedDAG(sn1), sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6

'''
                 +                        root
         /.6           .4\ 
        x                 x               pn1, pn2
    /      \          /      \ 
   +        +         +       +           cn1, cn2, cn3, cn4 
.1/ \.9  .3/ \.7   .2/ \.8 .4/ \.6
   X        Y         X       Y
'''
def myrdag():
    # WARNING: do not change this code; many unit tests (including ones not in this module) rely on it being just so.
    dag = spnss.rdag.simple( [DiscreteRV(2), DiscreteRV(2)] )
    root = dag.root
    pn1, pn2 = root.children
    cn1, cn2 = pn1.children
    cn3, cn4 = pn2.children
    root.set_weights([0.6, 0.4])
    cn1.set_masses([0.1, 0.9])
    cn2.set_masses([0.3, 0.7])
    cn3.set_masses([0.2, 0.8])
    cn4.set_masses([0.4, 0.6])
    return dag, root, pn1, pn2, cn1, cn2, cn3, cn4


def test_parents():
    print '\bp.',
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    pars = dag.parents()
    assert pars[cn1] == [pn1]
    assert pars[cn2] == [pn1]
    assert pars[cn3] == [pn2]
    assert pars[cn4] == [pn2]
    assert pars[pn1] == [root]
    assert pars[pn2] == [root]
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    pars = dag.parents()
    assert pars[pn4] == [sn2]
    assert set(pars[pn5]) == set([sn2, sn3])
    assert set(pars[pn6]) == set([sn2, pn3])
    assert set(pars[sn2]) == set([pn1, pn2])
    assert set(pars[sn3]) == set([pn1, pn3])
    assert pars[pn1] == pars[pn2] == pars[pn3] == [sn1]
    assert pars[sn1] == []

def test_upward_closure():
    print '\buc.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    pars = dag.parents()
    assert spnss.rdag.upward_closure([pn4, pn5, pn6], pars) == set([sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6])
    assert spnss.rdag.upward_closure([pn4], pars) == set([sn1, sn2, pn1, pn2, pn4])
    assert spnss.rdag.upward_closure([pn5], pars) == set([sn1, sn2, sn3, pn1, pn2, pn3, pn5])
    assert spnss.rdag.upward_closure([pn6], pars) == set([sn1, sn2, pn1, pn2, pn3, pn6])
    assert spnss.rdag.upward_closure([sn2, sn3], pars) == set([sn1, sn2, sn3, pn1, pn2, pn3])
    assert spnss.rdag.upward_closure([sn2, pn2], pars) == set([sn1, sn2, pn1, pn2])
    assert spnss.rdag.upward_closure([sn1], pars) == set([sn1])
    assert spnss.rdag.upward_closure([sn1, pn1, pn2, pn3], pars) == set([sn1, pn1, pn2, pn3])

def test_downward_closure():
    print '\bdc.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    assert spnss.rdag.downward_closure([sn1]) == set([sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6])
    assert spnss.rdag.downward_closure([pn1, pn3]) == set([sn2, sn3, pn1, pn3, pn4, pn5, pn6])
    assert spnss.rdag.downward_closure([pn3]) == set([sn3, pn3, pn5, pn6])
    assert spnss.rdag.downward_closure([sn2, pn3]) == set([pn3, sn2, sn3, pn4, pn5, pn6])
    assert spnss.rdag.downward_closure([sn3, pn5]) == set([sn3, pn5])

def test_check_topo_seeds():
    print '\bcts.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    nodes = [sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6]
    pars = dag.parents()
    for n in nodes:
         assert spnss.rdag.check_topo_seeds([n])
         assert spnss.rdag.check_topo_seeds([n], pars)
    assert spnss.rdag.check_topo_seeds([pn1, pn2, pn3])
    assert spnss.rdag.check_topo_seeds([pn1, pn2, pn3], pars)
    assert spnss.rdag.check_topo_seeds([sn2, sn3])
    assert spnss.rdag.check_topo_seeds([sn2, sn3], pars)
    assert spnss.rdag.check_topo_seeds([pn4, pn5, pn6])
    assert spnss.rdag.check_topo_seeds([pn4, pn5, pn6], pars)
    for n in nodes:
        for d in spnss.rdag.descendants([n]):
            assert not spnss.rdag.check_topo_seeds([n, d])
        for a in spnss.rdag.ancestors([n], pars):
            assert not spnss.rdag.check_topo_seeds([n, a], pars)

def test_topo_sort():
    print '\bfts.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    assert_raises(ValueError, spnss.rdag.topo_sort, [sn1, pn1])
    t1 = spnss.rdag.topo_sort([sn2, pn3])
    assert t1.index(sn2) in [0,1,2]
    assert t1.index(pn3) in [0,1,2]
    assert t1.index(sn3) in [1,2,3,4]
    assert t1.index(pn4) in range(1,6)
    assert t1.index(pn5) in range(3,6)
    assert t1.index(pn6) in range(2,6)
    t2 = spnss.rdag.topo_sort([pn2])
    assert t2[:2] == [pn2, sn2]
    assert set(t2[2:]) == set([pn4, pn5, pn6])
    t3 = spnss.rdag.topo_sort([pn1, pn3])
    assert t3.index(pn1) < t3.index(sn2) and t3.index(pn1) < t3.index(sn3)
    assert t3.index(sn2) < t3.index(pn4) and t3.index(sn2) < t3.index(pn5) and t3.index(sn2) < t3.index(pn6)
    assert t3.index(pn3) < t3.index(sn3) and t3.index(pn3) < t3.index(pn6)
    assert t3.index(sn3) < t3.index(pn5)
    t4 = spnss.rdag.topo_sort([pn3])
    assert t4[0] == pn3 and t4.index(sn3) < t4.index(pn5)

def test_rtopo_sort():
    print '\brts.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    pars = dag.parents()
    assert_raises(ValueError, spnss.rdag.rtopo_sort, [sn1, pn1], pars)
    t1 = spnss.rdag.rtopo_sort([sn2, pn3], pars)
    assert t1.index(sn2) < t1.index(pn1) and t1.index(sn2) < t1.index(pn2)
    assert t1.index(pn3) < t1.index(sn1)
    assert t1.index(pn1) < t1.index(sn1)
    assert t1.index(pn2) < t1.index(sn1)
    t2 = spnss.rdag.rtopo_sort([pn4], pars)
    assert t2[:2] == [pn4, sn2]
    assert set(t2[2:4]) == set([pn1, pn2])
    assert t2[-1] == sn1
    t3 = spnss.rdag.rtopo_sort([pn4, sn3], pars)
    assert t3.index(pn4) < t3.index(sn2)
    assert t3.index(sn2) < t3.index(pn1) and t3.index(sn2) < t3.index(pn2)
    assert t3.index(sn3) < t3.index(pn1) and t3.index(sn3) < t3.index(pn3)
    assert t3.index(sn1) == len(t3)-1
    assert spnss.rdag.rtopo_sort([sn1], pars) == [sn1]
    assert spnss.rdag.rtopo_sort([pn1], pars) == [pn1, sn1]

def test_pot_to():
    print '\bfpt.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    nodes = [sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6]
    for n in nodes:
        pot = spnss.rdag.pot_to(n)
        for i in range(len(pot)):
            assert set(pot[:i]) >= spnss.rdag.descendants([pot[i]])

def test_rpot_to():
    print '\brpt.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    nodes = [sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6]
    pars = dag.parents()
    for n in nodes:
        rpot = spnss.rdag.rpot_to(n, pars)
        for i in range(len(rpot)):
            assert set(rpot[:i]) >= spnss.rdag.ancestors([rpot[i]], pars)

def dag_forward(dag, X, Y, xval, yval):
    X.value = xval
    Y.value = yval
    for n in dag.pot():
        n.forward_log()

def test_max_downward_closure_of():
    print '\bmdco.',
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    X, Y = cn1.rv, cn2.rv
    dag_forward(dag, X, Y, 0, 0); assert pn2 in spnss.rdag.max_downward_closure_of(root)
    dag_forward(dag, X, Y, 0, 1); assert pn2 in spnss.rdag.max_downward_closure_of(root)
    dag_forward(dag, X, Y, 1, 0); assert pn2 in spnss.rdag.max_downward_closure_of(root)
    dag_forward(dag, X, Y, 1, 1); assert pn1 in spnss.rdag.max_downward_closure_of(root)

def test_max_weighted_downward_closure_of():
    print '\bmwdco.',
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    X, Y = cn1.rv, cn2.rv
    foo = spnss.rdag.max_weighted_downward_closure_of
    dag_forward(dag, X, Y, 0, 0); wdc = foo(root, True); assert set([root, pn2, cn3, cn4]) == wdc
    dag_forward(dag, X, Y, 0, 1); wdc = foo(root, True); assert set([root, pn2, cn3, cn4]) == wdc
    dag_forward(dag, X, Y, 1, 0); wdc = foo(root, True); assert set([root, pn1, cn1, cn2]) == wdc
    dag_forward(dag, X, Y, 1, 1); wdc = foo(root, True); assert set([root, pn1, cn1, cn2]) == wdc

def test_max_weighted_downward_closure_edges_of():
    print '\bmwdceo.',
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    X, Y = cn1.rv, cn2.rv
    foo = spnss.rdag.max_weighted_downward_closure_edges_of
    dag_forward(dag, X, Y, 0, 0); wdc = foo(root, True); assert set([(root,1), (pn2,0), (pn2,1), (cn3,0), (cn4,0)]) == wdc
    dag_forward(dag, X, Y, 0, 1); wdc = foo(root, True); assert set([(root,1), (pn2,0), (pn2,1), (cn3,0), (cn4,1)]) == wdc
    dag_forward(dag, X, Y, 1, 0); wdc = foo(root, True); assert set([(root,0), (pn1,0), (pn1,1), (cn1,1), (cn2,0)]) == wdc
    dag_forward(dag, X, Y, 1, 1); wdc = foo(root, True); assert set([(root,0), (pn1,0), (pn1,1), (cn1,1), (cn2,1)]) == wdc

def test_max_weighted_downward_closure_edges_of_v2():
    print '\bmwdceo2.',
    dag, root, pn1, pn2, cn1, cn2, cn3, cn4 = myrdag()
    X, Y = cn1.rv, cn2.rv
    foo = spnss.rdag.max_weighted_downward_closure_edges_of_v2
    dag_forward(dag, X, Y, 0, 0); wdc = foo(root, True); assert {root:1, pn2:-1, cn3:0, cn4:0} == wdc
    dag_forward(dag, X, Y, 0, 1); wdc = foo(root, True); assert {root:1, pn2:-1, cn3:0, cn4:1} == wdc
    dag_forward(dag, X, Y, 1, 0); wdc = foo(root, True); assert {root:0, pn1:-1, cn1:1, cn2:0} == wdc
    dag_forward(dag, X, Y, 1, 1); wdc = foo(root, True); assert {root:0, pn1:-1, cn1:1, cn2:1} == wdc

def test_has_cycle():
    print '\bhc.',
    dag, sn1, sn2, sn3, pn1, pn2, pn3, pn4, pn5, pn6 = dummydag()
    assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(sn1); assert spnss.rdag.has_cycle(sn1); pn6.remove_child(sn1); assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(pn1); assert spnss.rdag.has_cycle(sn1); pn6.remove_child(pn1); assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(pn2); assert spnss.rdag.has_cycle(sn1); pn6.remove_child(pn2); assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(pn3); assert spnss.rdag.has_cycle(sn1); pn6.remove_child(pn3); assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(sn2); assert spnss.rdag.has_cycle(sn1); pn6.remove_child(sn2); assert not spnss.rdag.has_cycle(sn1)
    pn6.add_child(sn3); assert not spnss.rdag.has_cycle(sn1); pn6.remove_child(sn3)


