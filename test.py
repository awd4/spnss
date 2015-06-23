import numpy as np

import spnss.test.test_nodes as nodestest
import spnss.test.test_vv as vvtest
import spnss.test.test_rdag as rdagtest
import spnss.test.test_network as networktest
import spnss.test.test_train as traintest
import spnss.test.test_cc as cctest
import spnss.test.test_operators as operatorstest

if True:
    print 'testing spnss.nodes... ',
    nodestest.test_Node()
    nodestest.test_DiscreteRV()
    nodestest.test_ContinuousRV()
    nodestest.test_GaussianNode()
    nodestest.test_CategoricalNode()
    nodestest.test_IndicatorNode()
    nodestest.test_SumNode()
    nodestest.test_ProdNode()
    nodestest.test_copy()
    nodestest.test_remove_children()
    print 'done.'
if True:
    print 'testing spnss.vv... ',
    vvtest.test_vvec()
    print 'done.'
if True:
    print 'testing spnss.rdag... ',
    rdagtest.test_parents()
    rdagtest.test_upward_closure()
    rdagtest.test_downward_closure()
    rdagtest.test_check_topo_seeds()
    rdagtest.test_topo_sort()
    rdagtest.test_rtopo_sort()
    rdagtest.test_pot_to()
    rdagtest.test_rpot_to()
    rdagtest.test_max_downward_closure_of()
    rdagtest.test_max_weighted_downward_closure_of()
    rdagtest.test_max_weighted_downward_closure_edges_of()
    rdagtest.test_max_weighted_downward_closure_edges_of_v2()
    rdagtest.test_has_cycle()
    print 'done.'
if True:
    print 'testing spnss.network... ',
    networktest.test_basic()
    networktest.test_scopes()
    networktest.test_pot()
    networktest.test_save_load()
    print 'done.'
if True:
    print 'testing spnss.train... ',
    traintest.test_categorical_node_map_with_dirichlet_prior()
    traintest.test_categorical_mle()
    traintest.test_initialize_network()
    traintest.test_backward_max_generative_trainer()
    traintest.test_backward_max_discriminative_gd_trainer()
    print 'done.'
if True:
    print 'testing spnss.cc... ',
    cctest.test_components()
    print 'done.'
if True:
    print 'testing spnss.operators... ',
    operatorstest.test_history()
    print 'done.'


