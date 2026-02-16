# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t, uint16_t, uint64_t
cimport cython

cdef extern from "ManyBodyState.cpp":
    pass

cdef extern from "ManyBodyState.h":
    cdef cppclass ManyBodyState:

        ctypedef vector[uint64_t] Key
        ctypedef vector[Key] Keys
        ctypedef vector[cython.doublecomplex] Values

        cppclass KeyComparer:
            bint operator()(const Key&, const Key&)
        cppclass ValueComparer:
            bint operator()(const cython.doublecomplex&, const cython.doublecomplex&)
        # ctypedef Map.iterator iterator
        # ctypedef Map.const_iterator const_iterator
        # ctypedef Map.reverse_iterator reverse_iterator
        # ctypedef Map.const_reverse_iterator const_reverse_iterator
        ctypedef size_t size_type 
        ctypedef pair[const Key, cython.doublecomplex] value_type
        ctypedef Key key_type

        ctypedef map[Key, cython.doublecomplex].iterator iterator
        ctypedef map[Key, cython.doublecomplex].const_iterator const_iterator

        ManyBodyState() nogil
        ManyBodyState(const ManyBodyState&) nogil
        ManyBodyState& operator=(const ManyBodyState&) nogil
        ManyBodyState(const Keys&,
                      const Values&) nogil

        bint empty() nogil
        size_type size() nogil
        size_type max_size() nogil
        void clear() nogil
        void prune(double) nogil
        double norm2() nogil
        double norm() nogil

        cython.doublecomplex& operator[](const key_type&) nogil
        cython.doublecomplex& at(const key_type&) nogil

        # ManyBodyState operator+=(const ManyBodyState&)
        # ManyBodyState operator-=(const ManyBodyState&)
        # ManyBodyState operator*=(const ManyBodyState&)
        # ManyBodyState operator/=(const ManyBodyState&)
        ManyBodyState operator-() nogil
        bint operator==(const ManyBodyState&) nogil
        bint operator!=(const ManyBodyState&) nogil
        iterator begin() nogil
        iterator end() nogil
        const_iterator cbegin() nogil
        const_iterator cend() nogil

        pair[iterator, bint] insert(const value_type&) nogil
        iterator insert(iterator, const value_type&) nogil
        void insert[InputIt](InputIt, InputIt) nogil


        iterator erase(iterator) nogil
        iterator erase(const_iterator) nogil
        iterator erase(const_iterator, const_iterator) nogil
        iterator erase(const key_type&) nogil

        void swap(ManyBodyState&) nogil

        iterator find[K](const K&) nogil

        iterator lower_bound[K](const K&) nogil

        iterator upper_bound[K](const K&) nogil
        ManyBodyState operator*(cython.doublecomplex, const ManyBodyState&) nogil
        ManyBodyState operator+(const ManyBodyState&, const ManyBodyState&) nogil
        ManyBodyState operator-(const ManyBodyState&, const ManyBodyState&) nogil
        ManyBodyState operator*(const ManyBodyState&, cython.doublecomplex) nogil
        ManyBodyState operator/(const ManyBodyState&, cython.doublecomplex) nogil

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&) nogil
