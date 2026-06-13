# distutils: language = c++

from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint16_t, uint64_t
from libcpp.string cimport string
cimport cython

cdef extern from "SlaterDeterminant.h" nogil:
    cdef cppclass SlaterDeterminant[CHUNK](vector[CHUNK]):
        # SlaterDeterminant()
        # SlaterDeterminant(const SlaterDeterminant&)
        # SlaterDeterminant& operator=(const SlaterDeterminant&)
        size_t hash()
        string to_string()
