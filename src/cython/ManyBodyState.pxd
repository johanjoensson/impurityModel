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

        ManyBodyState()
        ManyBodyState(const ManyBodyState&)
        ManyBodyState& operator=(const ManyBodyState&)
        ManyBodyState(const vector[key_type]&,
                      const vector[doublecomplex]&)

        bint empty()
        size_type size()
        size_type max_size()
        void clear()
        void prune(double)
        double norm2()
        double norm()

        double complex& operator[](const key_type&)
        double complex& at(const key_type&)

        # ManyBodyState operator+=(const ManyBodyState&)
        # ManyBodyState operator-=(const ManyBodyState&)
        # ManyBodyState operator*=(const ManyBodyState&)
        # ManyBodyState operator/=(const ManyBodyState&)
        ManyBodyState operator+(const ManyBodyState&)
        ManyBodyState operator-(const ManyBodyState&)
        ManyBodyState operator-()
        ManyBodyState operator*(double complex)
        ManyBodyState operator/(double complex)
        bint operator==(const ManyBodyState&)
        bint operator!=(const ManyBodyState&)
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
        iterator erase(const_iterator)
        iterator erase(const_iterator, const_iterator)
        size_type erase(const key_type&)

        void swap(ManyBodyState&)

        size_type count[K](const K&)

        iterator find[K](const K&)

        iterator lower_bound[K](const K&)

        iterator upper_bound[K](const K&)

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&)
    cdef ManyBodyState operator*(cython.doublecomplex, const ManyBodyState&)
