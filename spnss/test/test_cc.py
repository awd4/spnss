
import numpy as np

from spnss.cc import components

def test_components():
    def ctest(am):
        # am - adjacency matrix
        def test(i ,j):
            return am[i,j] == 1
        return test
    am = np.array([[0,1,1,0,0,0,0],[1,0,1,0,0,0,0],[1,1,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,1,0,1,1],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0]])
    cl = components(ctest(am), range(len(am)))
    cls = [set(c) for c in cl]
    assert len(cl) == 2 and set([0,1,2]) in cls and set([3,4,5,6]) in cls


