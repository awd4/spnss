import random

import spnss.network as network
import spnss.operators as ops


#n = network.independent('4d3')
#n = network.simple('8d4')

def test_history():
    print '\bh.',
    net = network.independent('8d3')

    def do_random_op():
        # randomly select a product node and child subset
        pnl = [n for n in net.pot if n.is_prod() and len(n.children) >=2]
        pn = random.choice(pnl)
        chl_size = random.choice(range(2, len(pn.children)+1))
        chl = list(pn.children)
        random.shuffle(chl)
        chl = chl[:chl_size]

        # perform an operation
        mo = ops.MixOp(net, pn, chl)
        mo.connect(random.choice(range(2, 5)))
        net.pot = None
        return mo

    opot = list(net.pot)
    ops_hist = []
    pot_hist = []
    for i in range(10):
        ops_hist.append( do_random_op() )
        pot_hist.append( list(net.pot) )
    for i in range(len(pot_hist))[::-1]:
        assert len(pot_hist[i]) == len(net.pot)
        assert set(pot_hist[i]) == set(net.pot)
        ops_hist[i].undo()
    assert len(opot) == len(net.pot)
    assert set(opot) == set(net.pot)
    

