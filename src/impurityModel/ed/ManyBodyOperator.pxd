# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int64_t
cimport cython

from ManyBodyState cimport ManyBodyState

cdef extern from "ManyBodyOperator.cpp":
    pass

cdef extern from "ManyBodyOperator.h":
    cdef cppclass ManyBodyOperator:
        ManyBodyOperator()
        ManyBodyOperator(const ManyBodyOperator&)
        ManyBodyOperator& operator=(const ManyBodyOperator&)
        ManyBodyOperator(const vector[pair[vector[int64_t], cython.doublecomplex]]&)

        void add_ops(const vector[pair[vector[int64_t], cython.doublecomplex]]&)
        
        double complex& operator[](const vector[int64_t]&)
        double complex& at(const vector[int64_t]&)
        ManyBodyState operator()(const ManyBodyState&, double)

        void clear_memory()

        ManyBodyOperator operator-()
        ManyBodyOperator operator+(const ManyBodyOperator&)
        ManyBodyOperator operator-(const ManyBodyOperator&)
        ManyBodyOperator operator*(const double complex&)
        ManyBodyOperator operator/(const double complex&)

        bint empty()
        size_t size()
        size_t max_size()
        void clear()

        bint operator==(const ManyBodyOperator&)
        bint operator!=(const ManyBodyOperator&)
        cppclass iterator:
            pair[vector[int64_t], doublecomplex] operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass const_iterator:
            pair[vector[int64_t], doublecomplex] operator*()
            iterator operator++()
            bint operator==(const_iterator)
            bint operator!=(const_iterator)
        iterator begin()
        iterator end()
        const_iterator cbegin()
        const_iterator cend()

        pair[iterator, bint] insert(const pair[vector[int64_t], doublecomplex]&)
        iterator insert(iterator, const pair[vector[int64_t], doublecomplex]&)
        void insert[InputIt](InputIt, InputIt)

        iterator erase(iterator)
        size_t erase(const vector[int64_t]&)

        void swap(ManyBodyOperator&)

        size_t count[K](const K&)

        iterator find[K](const K&)

        iterator lower_bound[K](const K&)

        iterator upper_bound[K](const K&)
