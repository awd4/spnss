# distutils: language = c++

'''
    Class Hierarchy:

    Node <--+<-- SumNode
            |
            +<-- ProdNode
            |
            +<-- DiscreteRV
            |
            +<-- ContinuousRV
            |
            +<-- GaussianNode
            |
            +<-- CategoricalNode
            | 
            +<-- IndicatorNode

    The GuassianNode, CategoricalNode, and IndicatorNode represent distributions
    over random variables. These nodes thus have a single child node representing
    that random variable: either a DiscreteRV or ContinuousRV. An IndicatorNode
    is like a CategoricalNode that places all its probability mass on a single
    value of the random variable; IndicatorNodes are the equivalent of the leaf
    nodes found in Poon's SPNs.

    A SumNode computes the dot product of its weight vector and the values of
    its child nodes. A SumNode may not have a random variable node as a child.

    A ProdNode computes the product of its children. It also may not have a
    random variable node as a child.
'''

from libcpp.vector cimport vector

cimport spnss.mathutil as mu


cdef vector[double] workspace_


cdef void sp_add_child(list children, Node child) except*:
    if child in children:
        raise ValueError, 'add_child(): node already in the children list'
    if not(child.is_sp() or child.is_distribution()):
        raise ValueError, 'add_child(): cannot add rv to sum/product node'
    children.append(child)


def to_data(n):
    if   n.is_sum():            return ['sum',  n.children, list(n.weights)]
    elif n.is_prod():           return ['prod', n.children]
    elif n.is_discrete():       return ['discrete',   n.name, n.num_values]
    elif n.is_continuous():     return ['continuous', n.name, n.support]
    elif n.is_gaussian():       return ['gaussian',    n.rv, n.mean, n.stdev]
    elif n.is_categorical():    return ['categorical', n.rv, list(n.masses)]
    elif n.is_indicator():      return ['indicator',   n.rv, n.indicator_value]
    else:
        raise Exception, 'Unknown node type.'

def from_data(data):
    d = data
    if   d[0] == 'sum':
        s = SumNode(); s.add_children(d[1]); s.set_weights(d[2]); return s
    elif d[0] == 'prod':
        p = ProdNode(); p.add_children(d[1]); return p
    elif d[0] == 'discrete':
        return DiscreteRV( d[2], d[1] )
    elif d[0] == 'continuous':
        return ContinuousRV( d[2], d[1] )
    elif d[0] == 'gaussian':
        g = GaussianNode( d[2], d[3] ); g.set_rv(d[1]); return g
    elif d[0] == 'categorical':
        c = CategoricalNode(); c.set_rv(d[1]); c.set_masses(d[2]); return c
    elif d[0] == 'indicator':
        i = IndicatorNode(d[2]); i.set_rv(d[1]); return i
    else:
        raise Exception, 'Invalid string.'



cdef class Node:

    def __cinit__(self):
        self.ntypes = 0
        self.value = 0.0

    cpdef bint is_sp(self):             return self.ntypes & NODE_SP
    cpdef bint is_sum(self):            return self.ntypes & NODE_SUM
    cpdef bint is_prod(self):           return self.ntypes & NODE_PROD

    cpdef bint is_rv(self):             return self.ntypes & NODE_RV
    cpdef bint is_discrete(self):       return self.ntypes & NODE_DISCRETE
    cpdef bint is_continuous(self):     return self.ntypes & NODE_CONTINUOUS

    cpdef bint is_distribution(self):   return self.ntypes & NODE_DISTRIBUTION
    cpdef bint is_gaussian(self):       return self.ntypes & NODE_GAUSSIAN
    cpdef bint is_categorical(self):    return self.ntypes & NODE_CATEGORICAL
    cpdef bint is_indicator(self):      return self.ntypes & NODE_INDICATOR

    cpdef bint has_children_attr(self): return self.is_sp()

    def label(self):
        if   self.is_discrete():    return 'DRV'
        elif self.is_continuous():  return 'CRV'
        elif self.is_gaussian():    return 'G'
        elif self.is_categorical(): return 'C'
        elif self.is_indicator():   return str(self.indicator_value)
        elif self.is_sum():         return '+'
        elif self.is_prod():        return 'X'
        else:
            raise Exception, 'Unlabeled type encountered'

    def children_iter(self):
        cdef Node c
        if not self.has_children_attr():
            return
        for c in self.children:
            yield c




cdef class SumNode(Node):

    def __cinit__(self):
        self.ntypes = NODE_SP | NODE_SUM
        self.children = []
        self.weights = mu.dvector()

    def copy(self):
        cp = SumNode()
        cp.ntypes = self.ntypes
        for c in self.children:
            cp.add_child(c)
        cp.weights = self.weights.copy()
        return cp

    def add_child(self, Node child not None):
        sp_add_child(self.children, child)
        self.weights.push_back(0.0)

    def remove_child(self, Node child not None):
        cdef unsigned int i = self.children.index(child)
        self.weights.pop_at(i)
        self.children.remove(child)

    def add_children(self, children):
        for c in children:
            self.add_child(c)

    def remove_children(self, children):
        children = list(children) if id(children) == id(self.children) else children
        for c in children:
            self.remove_child(c)

    def clear_children(self):
        self.children = []
        self.weights.clear()

    def set_weights(self, new_weights):
        self.weights.set(new_weights)

    def normalize(self):
        if self.weights.sum() == 0.0:
            self.weights.set([1.]*self.weights.size())
        self.weights.normalize()

    cpdef double forward(self):
        cdef unsigned int i
        cdef vector[double] *wvec = self.weights.vec
        self.value = 0.0
        for i in range(wvec.size()):
            self.value += wvec[0][i] * (<Node>self.children[i]).value
        return self.value

    cpdef double forward_log(self):     # backward, backward_log
        cdef unsigned int i, wsize
        cdef vector[double] *wvec = self.weights.vec
        wsize = wvec.size()
        if workspace_.size() < wsize:
            workspace_.resize( wsize )
        for i in range(wsize):
            workspace_[i] = mu.log(wvec[0][i]) + (<Node>self.children[i]).value
        self.value = mu.logsumexp( workspace_, wsize )
        return self.value

    cpdef double forward_max(self):
        cdef unsigned int i
        cdef double v
        cdef vector[double] *wvec = self.weights.vec
        self.value = mu.NINF
        for i in range(wvec.size()):
            v = wvec[0][i] * (<Node>self.children[i]).value
            if v > self.value:
                self.value = v
        return self.value

    cpdef double forward_max_log(self):
        cdef unsigned int i
        cdef double v
        cdef vector[double] *wvec = self.weights.vec
        self.value = mu.NINF
        for i in range(wvec.size()):
            v = mu.log(wvec[0][i]) + (<Node>self.children[i]).value
            if v > self.value:
                self.value = v
        return self.value

    def max_child_index(self):
        return mu.argmax_fair([c.value for c in self.children])

    def max_child(self):
        return self.children[self.max_child_index()]

    def max_weighted_child_index(self, bint logspace):
        if logspace:
            cvals = [mu.log(self.weights[i]) + self.children[i].value for i in range(len(self.children))]
        else:
            cvals = [       self.weights[i]  * self.children[i].value for i in range(len(self.children))]
        return mu.argmax_fair(cvals)

    def max_weighted_child(self, bint logspace):
        return self.children[self.max_weighted_child_index(logspace)]



cdef class ProdNode(Node):
    
    def __cinit__(self):
        self.ntypes = NODE_SP | NODE_PROD
        self.children = []

    def copy(self):
        cp = ProdNode()
        cp.ntypes = self.ntypes
        for c in self.children:
            cp.add_child(c)
        return cp

    def add_child(self, Node child not None):
        sp_add_child(self.children, child)

    def remove_child(self, Node child not None):
        self.children.remove(child)

    def add_children(self, children):
        for c in children:
            self.add_child(c)

    def remove_children(self, children):
        children = list(children) if id(children) == id(self.children) else children
        for c in children:
            self.remove_child(c)

    def clear_children(self):
        self.children = []

    cpdef double forward(self):
        cdef unsigned int i
        self.value = 1.0
        for i in range(len(self.children)):
            self.value *= (<Node>self.children[i]).value
        return self.value
    
    cpdef double forward_log(self):
        cdef unsigned int i
        self.value = 0.0
        for i in range(len(self.children)):
            self.value += (<Node>self.children[i]).value
        return self.value



cdef class DiscreteRV(Node):

    def __cinit__(self):
        self.ntypes = NODE_RV | NODE_DISCRETE
        self.value = mu.NAN
 
    def __init__(self, int num_values=2, name=''):
        self.name = name
        if not(num_values in [-1, -2] or num_values >= 2):
            raise ValueError, 'DiscreteRV(): invalid values code.'
        self.num_values = num_values

    def value_in_support(self, double value):
        if   mu.isnan(value):       return True
        elif value != <int>value:   return False
        elif self.num_values == -1: return value >= 0   # natural numbers {0,1,2,...}
        elif self.num_values == -2: return True         # integers {...,-2,-1,0,1,2,...}
        else:                       return 0 <= value < self.num_values

    cpdef bint is_known(self):
        return not mu.isnan(self.value)



cdef class ContinuousRV(Node):

    def __cinit__(self):
        self.ntypes = NODE_RV | NODE_CONTINUOUS
        self.value = mu.NAN

    def __init__(self, support=None, name=''):
        self.name = name
        support = ((mu.NINF, mu.INF),) if support is None else tuple(support)
        if False in [l<=u for l,u in support]:
            raise ValueError, 'ContinuousRV(): invalid support intervals'
        self.support = support

    def value_in_support(self, double value):
        if mu.isnan(value):
            return True
        for l,u in self.support:
            if l <= value <= u:
                return True
        return False

    cpdef bint is_known(self):
        return not mu.isnan(self.value)



cdef class GaussianNode(Node):

    def __cinit__(self):
        self.ntypes = NODE_DISTRIBUTION | NODE_GAUSSIAN
        self.rv = None

    def __init__(self, double mean=0.0, double stdev=1.0):
        if not( mu.NINF < mean < mu.INF and stdev > 0.0 ):
            raise ValueError, 'GaussianNode(): non-finite mean or non-positive standard deviation'
        self.mean = mean
        self.stdev = stdev

    def copy(self):
        cp = GaussianNode()
        cp.ntypes = self.ntypes
        cp.rv = self.rv
        cp.mean = self.mean
        cp.stdev = self.stdev
        return cp

    def set_rv(self, ContinuousRV rv not None):
        assert rv.support == ((mu.NINF, mu.INF),)
        self.rv = rv

    cpdef double forward(self):
        if not self.rv.is_known():
            self.value = 1.0; return self.value
        cdef double xmm = self.rv.value - self.mean
        self.value = (1.0 / (self.stdev * mu.SQRT_TWO_PI) ) * mu.exp( -xmm*xmm / (2.0 * self.stdev * self.stdev) )
        return self.value

    cpdef double forward_log(self):
        if not self.rv.is_known():
            self.value = 0.0; return self.value
        cdef double xmm = self.rv.value - self.mean
        self.value = mu.log(1.0 / (self.stdev * mu.SQRT_TWO_PI) ) + ( -xmm*xmm / (2.0 * self.stdev * self.stdev) )
        return self.value



cdef class CategoricalNode(Node):

    def __cinit__(self):
        self.ntypes = NODE_DISTRIBUTION | NODE_CATEGORICAL
        self.rv = None
        self.masses = mu.dvector()

    def copy(self):
        cp = CategoricalNode()
        cp.ntypes = self.ntypes
        cp.rv = self.rv
        cp.masses = self.masses.copy()
        return cp

    def set_rv(self, DiscreteRV rv not None):
        if rv.num_values < 2:
            raise ValueError, 'CategoricalNode.set_rv(): categorical must be over finite-valued discrete rv'
        self.rv = rv
        # Does this line produce a memory leak? Should we need to "del self.masses.vec" first?
        self.masses.vec[0] = [1.0 / rv.num_values for i in range(rv.num_values)]

    def set_masses(self, new_masses):
        self.masses.set(new_masses)

    def normalize(self):
        if self.masses.sum() == 0.0:
            self.masses.set([1.]*self.masses.size())
        self.masses.normalize()

    cpdef double forward(self):
        if not self.rv.is_known():
            self.value = self.masses.sum(); return self.value
        cdef unsigned int i = <int>self.rv.value
        if not( 0 <= i < self.masses.size() ):
            self.value = mu.NAN
            return self.value   # this should raise an exception, but I can't get Cython to do that
        self.value = self.masses[i]
        return self.value

    cpdef double forward_log(self):
        self.value = mu.log(self.forward())
        return self.value



cdef class IndicatorNode(Node):

    def __cinit__(self):
        self.ntypes = NODE_DISTRIBUTION | NODE_INDICATOR
        self.rv = None

    def __init__(self, double indicator_value):
        if not(mu.NINF < indicator_value < mu.INF):
            raise ValueError, 'IndicatorNode(): invalid indicator value'
        self.indicator_value = indicator_value

    def copy(self):
        cp = IndicatorNode()
        cp.ntypes = self.ntypes
        cp.rv = self.rv
        cp.indicator_value = self.indicator_value
        return cp

    def set_rv(self, Node rv):
        assert rv.is_rv()
        assert rv.value_in_support( self.indicator_value )
        self.rv = rv

    cpdef double forward(self):
        if not self.rv.is_known():
            self.value = 1.0; return self.value
        self.value = 1.0 if self.indicator_value == self.rv.value else 0.0
        return self.value

    cpdef double forward_log(self):
        if not self.rv.is_known():
            self.value = 0.0; return self.value
        self.value = 0.0 if self.indicator_value == self.rv.value else mu.NINF
        return self.value



