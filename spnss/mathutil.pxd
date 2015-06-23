# distutils: language = c++

from libcpp.vector cimport vector

IF UNAME_SYSNAME == "Windows":
    cdef extern from 'math.h':
        double sqrt(double)
    from numpy import log1p
    cdef inline bint isnan(double x):
        return x != x
ELSE:
    from libc.math cimport sqrt, log1p
    cdef extern from 'math.h':
        bint isnan(double x)


cdef double NINF
cdef double INF
cdef double PI
cdef double TWO_PI
cdef double SQRT_TWO_PI
cdef double NAN


cpdef double log(double a)
cpdef double exp(double a)

cdef double logsumexp_pointer(double* vals, unsigned int size)

cpdef double sum(vector[double]& vals, int size=*)
cpdef double logsumexp(vector[double]& vals, int size=*)
cpdef double logaddexp(double a, double b)
cpdef double logsubexp(double a, double b) except ? 1./0.

cpdef normalize(vector[double]& vals)

cpdef double dot(vector[double]& a, vector[double]& b) except ? 1./0.
cpdef double logdotexp(vector[double]& a, vector[double]& b) except ? 1./0.

cpdef root2pot(root, set visited=*)

cpdef int argmax_fair( list vals ) except -1
cpdef int argmin_fair( list vals ) except -1

cdef class dvector:
    cdef vector[double] *vec
    cpdef unsigned int size(self)
    cpdef resize(self, int n, double val=*)
    cpdef unsigned int capacity(self)
    cpdef bint empty(self)
    cpdef reserve(self, int size)
    cpdef double at(self, int i)
    cpdef double front(self)
    cpdef double back(self)
    cpdef push_back(self, double val)
    cpdef pop_back(self)
    cpdef clear(self)
    # Methods not in std::vector
    cpdef double sum(self)
    cpdef double logsumexp(self)
    cpdef normalize(self)
    cpdef double dot(self, dvector other)
    cpdef double logdotexp(self, dvector other)
    cpdef pop_at(self, unsigned int i)
    cpdef int set(self, other) except*


