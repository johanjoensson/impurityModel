# distutils: language = c++

cimport cython
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t

cdef extern from "ManyBodyState.cpp":
    pass

cdef extern from "ManyBodyState.h":
    cdef cppclass ManyBodyState:
        cppclass Comparer:
            bint operator()(const vector[uint8_t]&, const vector[uint8_t]&)
        ctypedef map[vector[uint8_t], cython.doublecomplex, Comparer] Map
        ctypedef Map.key_type key_type
        ctypedef Map.value_type value_type
        ctypedef Map.size_type size_type 
        ctypedef Map.value_type& reference
        ctypedef const Map.value_type& const_reference
        ctypedef Map.iterator iterator
        ctypedef Map.const_iterator const_iterator
        ctypedef Map.reverse_iterator reverse_iterator
        ctypedef Map.const_reverse_iterator const_reverse_iterator
        ctypedef Map.key_compare key_compare

        ManyBodyState() nogil
        ManyBodyState(const ManyBodyState&) nogil
        ManyBodyState& operator=(const ManyBodyState&) nogil
        ManyBodyState(const vector[key_type]&,
                      const vector[doublecomplex]&) nogil

        bint empty() nogil
        size_type size() nogil
        size_type max_size() nogil
        void clear() nogil
        void prune(double) nogil
        double norm2() nogil
        double norm() nogil

        double complex& operator[](const key_type&) nogil
        double complex& at(const key_type&) nogil

        # ManyBodyState operator+=(const ManyBodyState&)
        # ManyBodyState operator-=(const ManyBodyState&)
        # ManyBodyState operator*=(const ManyBodyState&)
        # ManyBodyState operator/=(const ManyBodyState&)
        ManyBodyState operator+(const ManyBodyState&) nogil
        ManyBodyState operator-(const ManyBodyState&) nogil
        ManyBodyState operator-() nogil
        ManyBodyState operator*(double complex) nogil
        ManyBodyState operator/(double complex) nogil
        bint operator==(const ManyBodyState&) nogil
        bint operator!=(const ManyBodyState&) nogil
        iterator begin() nogil
        iterator end() nogil
        const_iterator cbegin() nogil
        const_iterator cend() nogil
        reverse_iterator rbegin() nogil
        reverse_iterator rend() nogil
        const_reverse_iterator crbegin() nogil
        const_reverse_iterator crend() nogil

        pair[iterator, bint] insert(const value_type&) nogil
        iterator insert(iterator, const value_type&) nogil
        void insert[InputIt](InputIt, InputIt) nogil


        iterator erase(iterator) nogil
        iterator erase(const_iterator) nogil
        iterator erase(const_iterator, const_iterator) nogil
        size_type erase(const key_type&) nogil

        void swap(ManyBodyState&) nogil

        size_type count[K](const K&) nogil

        iterator find[K](const K&) nogil

        iterator lower_bound[K](const K&) nogil

        iterator upper_bound[K](const K&) nogil

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&) nogil
    cdef ManyBodyState operator*(cython.doublecomplex, const ManyBodyState&) nogil
