# distutils: language = c++

from util import assert_raises

from spnss.vv import vvec

def test_vvec():
    print '\bvv.',
    vrs = vvec('d')
    assert len(vrs) == 1 and vrs[0].name=='' and vrs[0].num_values==2
    assert vrs.code() == '1d2'
    vrs = vvec('c')
    assert len(vrs) == 1 and vrs[0].name=='' and vrs[0].support==((float('-inf'),float('inf')),)
    assert vrs.code() == '1c((-inf, inf),)'
    vrs = vvec('4c')
    assert len(vrs) == 4
    vrs = vvec('4d')
    assert len(vrs) == 4
    assert vrs.code() == '1d2*1d2*1d2*1d2'
    vrs = vvec('4c((1,2), (3,  4 ) )x1*d*d*2d*cx2')
    assert len(vrs) == 9
    assert False not in [v.support == ((1.,2.),(3.,4.)) for v in vrs[:4]]
    assert False not in [v.name == 'x1' for v in vrs[:4]]
    assert False not in [v.name == '' and v.num_values == 2 for v in vrs[4:8]]
    assert vrs[-1].name == 'x2'
    vrs = vvec('d52y')
    assert vrs[0].num_values == 52 and vrs[0].name == 'y'
    assert vrs.code() == '1d52y'


