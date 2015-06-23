# distutils: language = c++

from spnss.nodes cimport Node

# Helpers
#cpdef topo_sort(nodes) except*
#cpdef rtopo_sort(nodes, dict parents)
cpdef pot_to(Node n)
cpdef rpot_to(Node n, dict parents) 
cpdef max_downward_closure_of(Node n, set mdo=*)
cpdef max_weighted_downward_closure_of(Node n, bint logspace=*, set mdo=*)
cpdef max_weighted_downward_closure_edges_of(Node n, bint logspace=*, set mdo=*)
cpdef max_weighted_downward_closure_edges_of_v2(Node n, bint logspace=*, dict mdo=*)

# RootedDAG constructors
#def independent(vvec)


cdef class RootedDAG:
    cdef public Node            root


