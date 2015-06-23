# distutils: language = c++

from libcpp.vector cimport vector

cimport spnss.mathutil as mu


cdef enum:
    NODE_SP = 1                 # SP nodes
    NODE_SUM = 2
    NODE_PROD = 4

    NODE_RV = 8                 # Random variable nodes
    NODE_DISCRETE = 16
    NODE_CONTINUOUS = 32

    NODE_DISTRIBUTION = 64      # Distribution nodes
    NODE_GAUSSIAN = 128 
    NODE_CATEGORICAL = 256
    NODE_INDICATOR = 512

    DISCRETE_RV_NATURALS = -1
    DISCRETE_RV_INTEGERS = -2


cdef void sp_add_child(list children, Node child) except*


cdef class Node:
    cdef readonly int               ntypes
    cdef public double              value

    cpdef bint is_sp(self)
    cpdef bint is_sum(self)
    cpdef bint is_prod(self)

    cpdef bint is_rv(self)
    cpdef bint is_discrete(self)
    cpdef bint is_continuous(self)

    cpdef bint is_distribution(self)
    cpdef bint is_gaussian(self)
    cpdef bint is_categorical(self)
    cpdef bint is_indicator(self)

    cpdef bint has_children_attr(self)



cdef class SumNode(Node):
    cdef public list                children
    cdef public mu.dvector          weights
    cpdef double forward(self)
    cpdef double forward_log(self)
    cpdef double forward_max(self)
    cpdef double forward_max_log(self)



cdef class ProdNode(Node):
    cdef public list                children
    cpdef double forward(self)
    cpdef double forward_log(self)



cdef class DiscreteRV(Node):
    cdef public str                 name
    cdef public int                 num_values  # -1 means any number in {0, 1, 2, 3, ... }, -2 means {...,-2,-1,0,1,2,...}
    cpdef bint is_known(self)
 


cdef class ContinuousRV(Node):
    cdef public str                 name
    cdef public tuple               support     # a set of intervals describing the values it can take
    cpdef bint is_known(self)



cdef class GaussianNode(Node):
    cdef public ContinuousRV        rv
    cdef public double              mean
    cdef public double              stdev
    cpdef double forward(self)
    cpdef double forward_log(self)



cdef class CategoricalNode(Node):
    cdef public DiscreteRV          rv
    cdef public mu.dvector          masses
    cpdef double forward(self)
    cpdef double forward_log(self)



cdef class IndicatorNode(Node):
    cdef public Node                rv
    cdef public double              indicator_value # value of child RV that causes this node to take the value one
    cpdef double forward(self)
    cpdef double forward_log(self)



