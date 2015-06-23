# distutils: language = c++

from math import pi
import random
import numpy as np

from libc.math cimport log as c_log
from libc.math cimport exp as c_exp


NINF = float('-inf')
INF  = float('inf')
PI = pi
TWO_PI = 2.0 * PI
SQRT_TWO_PI = sqrt( TWO_PI )
NAN = float('nan')


cpdef double log(double a):
    if a == 0.0:
        return NINF
    return c_log(a)


cpdef double exp(double a):
    return c_exp(a)


cpdef double sum(vector[double]& vals, int size=-1):
    cdef unsigned int i, vsize
    cdef double v = 0.0
    vsize = vals.size() if size < 0 else size
    for i in range(vsize):
        v += vals[i]
    return v
    

cdef double logsumexp_pointer(double* vals, unsigned int size):
    cdef unsigned int i
    cdef double value = 0.0
    cdef double max_value = NINF
    for i in range(size):
        if vals[i] > max_value:
            max_value = vals[i]
    if max_value == NINF:
        return NINF
    for i in range(size):
        value += exp( vals[i] - max_value )
    return max_value + log( value )


cpdef double logsumexp(vector[double]& vals, int size=-1):
    cdef unsigned int vsize
    vsize = vals.size() if size < 0 else size
    return logsumexp_pointer(&vals[0], vsize)
#    cdef unsigned int i, vsize
#    cdef double value = 0.0
#    cdef double max_value = NINF
#    vsize = vals.size() if size < 0 else size
#    for i in range(vsize):
#        if vals[i] > max_value:
#            max_value = vals[i]
#    if max_value == NINF:
#        return NINF
#    for i in range(vsize):
#        value += exp( vals[i] - max_value )
#    return max_value + log( value )


cpdef double logaddexp(double a, double b):
    # translated from NumPy source code
    cdef double tmp = a - b
    if tmp > 0:
        return a + log1p( exp(-tmp) )
    elif tmp <= 0:
        return b + log1p( exp(tmp) )
    else:
        return a + b    # NaNs or infinities of the same sign involved


cpdef double logsubexp(double a, double b) except ? 1./0.:
    # http://stackoverflow.com/questions/778047/we-know-log-add-but-how-to-do-log-subtract
    if a < b:
        raise ValueError, 'logsubexp(): cannot take the log of a negative number'
    if b == NINF:
        return a
    return a + log1p(-exp(b-a))


cpdef normalize(vector[double]& vals):
    cdef unsigned int i
    cdef double total = 0.0
    for i in range(vals.size()):
        total += vals[i]
    for i in range(vals.size()):
        vals[i] /= total


cpdef double dot(vector[double]& a, vector[double]& b) except ? 1./0.:
    if a.size() != b.size():
        raise ValueError, 'dot(): vector lengths do not match'
    cdef unsigned int i
    cdef double v = 0.0
    for i in range(a.size()):
        v += a[i] * b[i]
    return v


cpdef double logdotexp(vector[double]& a, vector[double]& b) except ? 1./0.:
    if a.size() != b.size():
        raise ValueError, 'logdotexp(): vector lengths do not match'
    cdef unsigned int i
    cdef vector[double] c
    for i in range(a.size()):
        c.push_back(a[i] + b[i])
    return logsumexp(c)


cpdef root2pot(root, set visited=None):
    if visited is None:
        visited = set()
    visited.add( root )
    pot = []
    for c in root.children:
        if c in visited:
            continue
        cpot = root2pot( c, visited )
        pot.extend( cpot )
    pot.append( root )
    return pot


def random_dag(min_num_dag_leaves, num_nodes, num_edges, single_root):
    assert num_nodes > 0
    assert 1 <= min_num_dag_leaves <= num_nodes
    assert num_nodes >= min_num_dag_leaves + 1
    assert num_edges >= num_nodes - 1
    assert num_edges <= num_nodes*(num_nodes-1) / 2     # should be even less than this to take into account the leaf nodes
    nl, nn, ne = min_num_dag_leaves, num_nodes, num_edges
    A = np.zeros((nn, nn))      # adjacency matrix
    # 1. We'll use the first nl rows for the leaf nodes, so leave them as zeros
    nre = ne    # number of remaining edges
    if single_root:
        # 2. Make sure only the last column has all zeros (to ensure a single root node)
        for j in range(nn-1):
            i = random.choice( range( max(j+1, nl), nn ) )
            A[i,j] = 1
            nre -= 1
    # 3. Add remaining edges
    if nre > 0:
        candidates = []
        for i in range(nl,nn):
            for j in range(i):
                if A[i,j] == 0:
                    candidates.append( (i,j) )
        edges = random.sample( candidates, nre ) if len(candidates) > nre else candidates
        for ei, ej in edges:
            A[ei,ej] = 1
            nre -= 1
    return A


cpdef int argmax_fair( list vals ) except -1:
    if len(vals) == 0:
        return -1
    cdef int i
    max_val = max(vals)
    return random.choice( [i for i in range(len(vals)) if vals[i] == max_val] )


cpdef int argmin_fair( list vals ) except -1:
    if len(vals) == 0:
        return -1
    cdef int i
    min_val = min(vals)
    return random.choice( [i for i in range(len(vals)) if vals[i] == min_val] )


def shuffled(list x not None):
    x = x[:]
    random.shuffle(x)
    return x


def dvector_iter(dvector dv):
    cdef int i = 0
    cdef int s = dv.vec[0].size()
    for i in range(s):
        yield dv.vec[0][i]

cdef class dvector:
    def __cinit__(self):
        self.vec = new vector[double]()
    def __dealloc__(self):
        del self.vec
    cpdef unsigned int size(self):                  return self.vec.size()
    cpdef resize(self, int n, double val=0.0):      self.vec.resize(n, val)
    cpdef unsigned int capacity(self):              return self.vec.capacity()
    cpdef bint empty(self):                         return self.vec.empty()
    cpdef reserve(self, int size):                  self.vec.reserve(size)
    cpdef double at(self, int i):                   return self.vec.at(i)
    cpdef double front(self):                       return self.vec.front()
    cpdef double back(self):                        return self.vec.back()
    cpdef push_back(self, double val):              self.vec.push_back(val)
    cpdef pop_back(self):                           self.vec.pop_back()
    cpdef clear(self):                              self.vec.clear()
    # Methods not in std::vector
    cpdef double sum(self):                         return sum(self.vec[0])
    cpdef double logsumexp(self):                   return logsumexp(self.vec[0])
    cpdef normalize(self):                          normalize(self.vec[0])
    cpdef double dot(self, dvector other):          return dot(self.vec[0], other.vec[0])
    cpdef double logdotexp(self, dvector other):    return logdotexp(self.vec[0], other.vec[0])
    cpdef pop_at(self, unsigned int i):
        cdef unsigned int j
        for j in range(i, self.vec.size() - 1):
            self.vec[0][j] = self.vec[0][j+1]
        self.vec.pop_back()
    cpdef int set(self, other) except*:
        cdef unsigned int l = len(other)
        if l != self.vec.size():
            raise ValueError, 'dvector.set(): size of sequences do not match'
        for i,nv in enumerate(other):
            self.vec[0][i] = nv
        return 0
    def copy(self):
        cp = dvector()
        for val in self:
            cp.push_back(val)
        return cp
    def __getitem__(self, int i):
        cdef int s = self.vec.size()
        if i >= s or i < -s:
            raise IndexError, 'dvector index out of range'
        if i < 0:
            i += s
        return self.vec[0][i]
    def __setitem__(self, int i, double val):
        cdef int s = self.vec.size()
        if i >= s or i < -s:
            raise IndexError, 'dvector index out of range'
        if i < 0:
            i += s
        self.vec[0][i] = val
    def __iter__(self):
        return dvector_iter(self)
    def __len__(self):
        return self.vec[0].size()
    def __str__(self):
        return str([e for e in self])






#from util import assert_raises
def assert_raises(exception, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exception:
        pass
    except Exception as e:
        assert False, 'Function ' + str(func) + ' raised an unexpected exception: ' + str(e)
    else:
        assert False, 'Function ' + str(func) + ' did not raise ' + str(exception)




def test_log():
    assert log(0.0) == NINF


def test_sum():
    cdef vector[double] v = range(6)
    assert sum(v) == 15.0


def test_logsumexp():
    cdef vector[double] v = [log(<double>i) for i in range(6)]
    assert np.allclose( log(15.0), logsumexp(v) )
    v.clear()
    for val in [NINF, NINF]: v.push_back( val )
    assert np.allclose( logsumexp(v), NINF )


def test_logaddexp():
    for i in range(6):
        assert np.allclose( logaddexp( log(<double>i), log(5.0-i) ), log(5.0) )
        assert logaddexp(log(<double>i), NINF) == log(<double>i)
        assert logaddexp(NINF, log(<double>i)) == log(<double>i)
    assert logaddexp(NINF, NINF) == NINF


def test_logsubexp():
    for i in range(6):
        assert np.allclose( logsubexp( log(5.0+i), log(<double>i) ), log(5.0) )
        assert logaddexp(log(<double>i), NINF) == log(<double>i)
    assert logaddexp(NINF, NINF) == NINF
    assert_raises(ValueError, logsubexp, NINF, log(1.0))


def test_normalize():
    cdef vector[double] v = range(5)
    normalize(v)
    assert np.allclose( v, [0.0, 0.1, 0.2, 0.3, 0.4] )
    assert np.allclose( sum(v), 1.0 )


def test_dot():
    cdef vector[double] a = range(6)
    cdef vector[double] b = range(1,7)
    cdef vector[double] c = range(2,7)
    assert dot(a, b) == 70.0
    assert_raises(ValueError, dot, a, c)


def test_logdotexp():
    cdef vector[double] a = [log(<double>i) for i in range(6)]
    cdef vector[double] b = [log(<double>i) for i in range(1,7)]
    cdef vector[double] c = [log(<double>i) for i in range(2,7)]
    assert np.allclose( log(70.0), logdotexp(a, b) )
    assert_raises(ValueError, logdotexp, a, c)


def test_root2pot():
    class tnode:
        def __init__(self):
            self.children = []
    r, c1, c2, l = tnode(), tnode(), tnode(), tnode()
    r.children = [c1, c2]
    for c in r.children: c.children.append( l )
    pot = root2pot(r)
    assert len(pot) == 4
    assert pot[-1] == r
    assert pot[0] == l
    assert False not in [c in pot[1:3] for c in r.children] == [True, True]

        
def test_random_dag():
    A = random_dag( 3, 4, 3, True )
    assert sum( A.sum(axis=1) == 0 ) == 3
    assert sum( A.sum(axis=0) == 0 ) == 1
    assert A.sum() == 3

    A = random_dag( 1, 4, 6, True )
    assert sum( A.sum(axis=1) == 0 ) == 1
    assert sum( A.sum(axis=0) == 0 ) == 1
    assert A.sum() == 6


def test_argmax_fair():
    e = []
    assert_raises(Exception, argmax_fair, e)
    l = [-1, 1, 1, 1, 0]
    i = argmax_fair(l)
    assert i in [1, 2, 3]


def test_argmin_fair():
    e = []
    assert_raises(Exception, argmin_fair, e)
    l = [1, -1, -1, -1, 0]
    i = argmin_fair(l)
    assert i in [1, 2, 3]


def test_dvector():
    dv = dvector()
    dv.vec[0] = range(10)
    assert str(dv) == str([float(i) for i in range(10)])
    assert dv.size() == 10
    assert len(dv) == 10
    assert not dv.empty()
    assert dv[0] == 0.0
    assert dv.at(0) == 0.0
    assert dv.front() == 0.0
    assert dv.back() == 9.0
    assert dv[9] == 9.0
    assert_raises(IndexError, dv.__getitem__, 10)
    assert dv[-10] == 0.0
    assert_raises(IndexError, dv.__getitem__, -11)
    assert_raises(IndexError, dv.__setitem__, 10, 0.0)
    assert_raises(IndexError, dv.__setitem__, -11, 0.0)
    dv.push_back(10)
    assert dv.back() == 10.0
    dv.pop_back()
    assert dv.back() == 9.0
    assert [e for e in dv] == range(10)
    dv.clear()
    assert dv.size() == 0
    assert len(dv) == 0
    assert_raises(IndexError, dv.__getitem__, 0)
    assert str(dv) == str([])


def test_dvector_ops():
    cdef dvector dv = dvector()
    dv.vec[0] = range(4)
    cdef dvector cp = dv.copy()
    assert cp.size() == dv.size()
    assert [v for v in cp] == [v for v in dv]
    assert_raises(ValueError, dv.set, range(5))
    assert_raises(ValueError, dv.set, [])
    dv.set(range(1,5))
    assert dv.sum() == 10.0
    dv.pop_at(2)
    assert dv.size() == 3
    assert dv.sum() == 7.0


