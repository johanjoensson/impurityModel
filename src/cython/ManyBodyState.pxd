# distutils: language = c++

from SlaterDeterminant cimport SlaterDeterminant
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.complex cimport complex
from libc.stdint cimport uint8_t, uint16_t, uint64_t
cimport cython

cdef extern from "ManyBodyState.cpp" nogil:
    pass

cdef extern from "ManyBodyState.h" nogil:
    cdef cppclass ManyBodyState:

        ctypedef unordered_map[SlaterDeterminant[uint64_t], complex[double]] Map
        ctypedef Map.key_type key_type
        ctypedef complex[double] mapped_type
        ctypedef pair[SlaterDeterminant[uint64_t], complex[double]] value_type
        # ctypedef Map.value_type value_type
        ctypedef Map.size_type size_type
        ctypedef Map.difference_type difference_type
        ctypedef Map.iterator iterator
        ctypedef Map.const_iterator const_iterator

        ManyBodyState()
        ManyBodyState(const ManyBodyState&)
        ManyBodyState& operator=(const ManyBodyState&)
        ManyBodyState(const vector[key_type]&,
                      const vector[mapped_type]&)

        bint empty()
        size_type size()
        size_type max_size()
        void clear()
        void prune(double)
        double norm2()
        double norm()

        cython.doublecomplex& operator[](const key_type&)
        cython.doublecomplex& at(const key_type&)

        ManyBodyState operator-()
        bint operator==(const ManyBodyState&)
        bint operator!=(const ManyBodyState&)
        iterator begin()
        iterator end()
        const_iterator cbegin()
        const_iterator cend()

        pair[iterator, bint] insert(const value_type&)
        iterator insert(iterator, const value_type&)
        void insert[InputIt](InputIt, InputIt)

        iterator erase(iterator)
        iterator erase(const_iterator)
        iterator erase(const_iterator, const_iterator)
        iterator erase(const key_type&)

        void swap(ManyBodyState&)

        iterator find[K](const K&)

        ManyBodyState operator*(mapped_type, const ManyBodyState&)
        ManyBodyState operator+(const ManyBodyState&, const ManyBodyState&)
        ManyBodyState operator-(const ManyBodyState&, const ManyBodyState&)
        ManyBodyState operator*(const ManyBodyState&, mapped_type)
        ManyBodyState operator/(const ManyBodyState&, mapped_type)

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&) nogil
