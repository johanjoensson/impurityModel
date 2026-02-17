# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int64_t, uint8_t
cimport cython

from ManyBodyState cimport ManyBodyState

cdef extern from "ManyBodyOperator.cpp" nogil:
    pass

cdef extern from "ManyBodyOperator.h" nogil:
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

        ManyBodyOperator() 
        ManyBodyOperator(const ManyBodyOperator&) 
        ManyBodyOperator& operator=(const ManyBodyOperator&) 
        ManyBodyOperator(const vector[value_type]&) 
        ManyBodyOperator(const OPS_VEC&, const SCALAR_VEC&) 

        cython.doublecomplex& operator[](const key_type&) 
        cython.doublecomplex& at(const key_type&) 
        ManyBodyState operator()(const ManyBodyState&, double, const restrictions&) 
        ManyBodyState operator()(const vector[ManyBodyState]&, double, const restrictions&) 
        ManyBodyState apply(const ManyBodyState&, double, const restrictions&) 
        ManyBodyState apply(const vector[ManyBodyState]&, double, const restrictions&) 



    
        # ManyBodyOperator operator+=(const ManyBodyOperator&)
        # ManyBodyOperator operator-=(const ManyBodyOperator&)
        # ManyBodyOperator operator*=(const cython.doublecomplex&)
        # ManyBodyOperator operator/=(const cython.doublecomplex&)
        ManyBodyOperator operator-() 
        ManyBodyOperator operator+(const ManyBodyOperator&) 
        ManyBodyOperator operator-(const ManyBodyOperator&) 
        ManyBodyOperator operator*(const cython.doublecomplex&) 
        ManyBodyOperator operator/(const cython.doublecomplex&) 

        bint empty() 
        size_type size() 
        size_type max_size() 
        void clear() 

        bint operator==(const ManyBodyOperator&) 
        bint operator!=(const ManyBodyOperator&) 
        iterator begin() 
        iterator end() 
        const_iterator cbegin() 
        const_iterator cend() 
        reverse_iterator rbegin() 
        reverse_iterator rend() 
        const_reverse_iterator crbegin() 
        const_reverse_iterator crend() 

        pair[iterator, bint] insert(const value_type&) 
        iterator insert(iterator, const value_type&) 
        void insert[InputIt](InputIt, InputIt) 

        iterator erase(iterator) 
        iterator erase(const_iterator, const_iterator) 
        size_t erase(const key_type&) 

        void swap(ManyBodyOperator&) 

        iterator find[K](const K&) 
        iterator find[K](iterator, iterator, const K &) 

        iterator lower_bound[K](const K&) 

        iterator upper_bound[K](const K&) 
