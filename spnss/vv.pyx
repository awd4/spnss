# distutils: language = c++

import re
from numpy import ndarray

from nodes import Node, DiscreteRV, ContinuousRV


def tuple_code(s):
    mo = re.match(r'(?i)(?P<num>\d*)(?P<type>[dc])(?P<spec>\d+|-\d+|(\(\s*(\([^)]+\)[^(]*)+\s*\))*)(?P<name>.*$)', s)
    if not mo:
        raise ValueError, 'vv.vvec(): invalid code string'
    num = 1 if mo.group('num') == '' else int(mo.group('num'))
    tp = mo.group('type')
    if tp == 'd':
        spec = 2 if mo.group('spec') == '' else int(mo.group('spec'))
    elif tp == 'c':
        if mo.group('spec') == '':
            spec = None
        else:
            spec = tuple(float(e) for e in re.split(r'[(),\s]', mo.group('spec')) if e != '')
            spec = tuple((spec[i],spec[i+1]) for i in range(0,len(spec),2))
    name = mo.group('name')
    return (num, tp, spec, name)

def vvec(code):
    ''' Creates a VariableVector from a code.
        
        The argument 'code' can be 1) a numpy array, 2) a string,
        or 3) an iterable containing strings and tuples.
        
        We translate 'code' into a list of tuples. Each tuple
        has 4 elements that describe how to build one or more
        variables. The tuple elements, in order, correspond to
        1) the number of variables to create, 2) the type of
        variable, 'd' for discrete and 'c' for continuous, 3) the
        variable specification, and 4) the variable's name.
        
        Variable specifications are as follows:
            continous variable specs options:
                0 -> default support (-inf, inf)
                (l1,u1,l2,u2,...,lk,uk) -> support ranges
            discrete variable spec options:
                -1 -> natural numbers are valid values
                -2 -> all integers are valid values
                number -> number of valid values

        If 'code' is a numpy array then none of the variables
        have names. Each value in the numpy array corresponds to
        a single variable. The values indicate the variable
        specifications: 0 indicates a default continous variable
        and non-zero values indicate discrete variables.

        If 'code' is an iterable containing string and tuples then
        the tuples are used as-is and the strings are converted to
        tuples. 

        If 'code' is a string then codes are created by splitting
        'code' on the separator character '*' and converting each
        resulting substring into a tuple.
    '''
    if type(code) == ndarray:
        codes = [(1,'d',int(c),'') if int(c) != 0 else (1,'c',None,'') for c in code.flat]
    elif type(code) in [str, unicode]:
        codes = [tuple_code(c) for c in code.split('*')]
    elif hasattr(code, '__iter__') and False not in [isinstance(c, Node) for c in code]:
        codes = code
    elif hasattr(code, '__iter__'):
        codes = [c if type(c) == tuple else tuple_code(c) for c in code]
    else:
        raise ValueError, 'vv.vvec(): invalid code type'
    vrs = VariableVector()
    for c in codes:
        if isinstance(c, Node):
            vrs.add_variable(c)
            continue
        for i in range(c[0]):
            rv = DiscreteRV(c[2], c[3]) if c[1] == 'd' else ContinuousRV(c[2], c[3])
            vrs.add_variable(rv)
    return vrs



cdef class VariableVector(list):

    def add_variable(self, v):
        if not v.is_rv() or v in self:
            raise ValueError, 'VariableVector(): cannot add non-rv, cannot have duplicate rvs'
        self.append(v)

    def set(self, values, check_support=True):
        if len(values) != len(self):
            raise ValueError, 'VariableVector.set(): values sequence length not equal to number of variables'
        for i in range(len(values)):
            var = self[i]
            val = values[i]
            if check_support and not var.value_in_support(val):
                raise ValueError, 'VariableVector.set(): value not in support of the variable'
            var.value = <double>val

    def num_instantiations(self):
        n = 1
        for v in self:
            n *= v.num_values
        return n

    def code(self):
        def v2str(v):
            return '1' + ('d'+str(v.num_values) if v.is_discrete() else 'c'+str(v.support)) + v.name
        c = ''.join([v2str(v)+'*' for v in self[:-1]])
        c += v2str(self[-1])
        return c


