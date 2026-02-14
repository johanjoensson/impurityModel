# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int64_t, uint8_t
cimport cython

from ManyBodyState cimport ManyBodyState

cdef extern from "ManyBodyOperator.cpp":
    pass

cdef extern from "ManyBodyOperator.h":
    cdef cppclass ManyBodyOperator:
        cppclass Comparer[T]:
            bint operator()(const vector[T]&, const vector[T]&)

        ctypedef vector[int64_t] OPS
        ctypedef vector[OPS] OPS_VEC 
        ctypedef vector[cython.doublecomplex] SCALAR_VEC 
        ctypedef const vector[int64_t] key_type
        ctypedef pair[vector[int64_t], cython.doublecomplex] value_type
        ctypedef vector[value_type].size_type size_type 
        ctypedef value_type& reference_type
        ctypedef const value_type& const_reference_type
        ctypedef value_type& reference
        ctypedef const value_type& const_reference
        ctypedef vector[value_type].iterator iterator
        ctypedef vector[value_type].const_iterator const_iterator
        ctypedef vector[value_type].reverse_iterator reverse_iterator
        ctypedef vector[value_type].const_reverse_iterator const_reverse_iterator
        ctypedef vector[pair[vector[size_t], pair[size_t, size_t]]] restrictions

        ManyBodyOperator() nogil
        ManyBodyOperator(const ManyBodyOperator&) nogil
        ManyBodyOperator& operator=(const ManyBodyOperator&) nogil
        ManyBodyOperator(const vector[value_type]&) nogil
        ManyBodyOperator(const OPS_VEC&, const SCALAR_VEC&) nogil

        cython.doublecomplex& operator[](const key_type&) nogil
        cython.doublecomplex& at(const key_type&) nogil
        ManyBodyState operator()(const ManyBodyState&, double, const restrictions&) nogil
        ManyBodyState operator()(const vector[ManyBodyState]&, double, const restrictions&) nogil
        ManyBodyState apply(const ManyBodyState&, double, const restrictions&) nogil
        ManyBodyState apply(const vector[ManyBodyState]&, double, const restrictions&) nogil



    
        # ManyBodyOperator operator+=(const ManyBodyOperator&)
        # ManyBodyOperator operator-=(const ManyBodyOperator&)
        # ManyBodyOperator operator*=(const cython.doublecomplex&)
        # ManyBodyOperator operator/=(const cython.doublecomplex&)
        ManyBodyOperator operator-() nogil
        ManyBodyOperator operator+(const ManyBodyOperator&) nogil
        ManyBodyOperator operator-(const ManyBodyOperator&) nogil
        ManyBodyOperator operator*(const cython.doublecomplex&) nogil
        ManyBodyOperator operator/(const cython.doublecomplex&) nogil

        bint empty() nogil
        size_type size() nogil
        size_type max_size() nogil
        void clear() nogil

        bint operator==(const ManyBodyOperator&) nogil
        bint operator!=(const ManyBodyOperator&) nogil
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
        iterator erase(const_iterator, const_iterator) nogil
        size_t erase(const key_type&) nogil

        void swap(ManyBodyOperator&) nogil

        iterator find[K](const K&) nogil
        iterator find[K](iterator, iterator, const K &) nogil

        iterator lower_bound[K](const K&) nogil

        iterator upper_bound[K](const K&) nogil
