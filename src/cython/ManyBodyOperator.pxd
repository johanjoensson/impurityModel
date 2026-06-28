# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int64_t, uint8_t
from libcpp.complex cimport complex
cimport cython

from ManyBodyState cimport ManyBodyState

cdef extern from "ManyBodyOperator.cpp" nogil:
    pass

cdef extern from "ManyBodyOperator.h" nogil:
    cdef cppclass ManyBodyOperator:
        cppclass Comparer[T]:
            bint operator()(const vector[T]&, const vector[T]&)

        ctypedef vector[int64_t] key_type
        ctypedef complex[double] mapped_type
        ctypedef pair[key_type, mapped_type] value_type
        ctypedef vector[value_type].size_type size_type
        ctypedef vector[value_type].iterator iterator
        ctypedef vector[value_type].const_iterator const_iterator
        ctypedef vector[value_type].reverse_iterator reverse_iterator
        ctypedef vector[value_type].const_reverse_iterator const_reverse_iterator
        ctypedef vector[pair[vector[size_t], pair[size_t, size_t]]] restrictions
        ctypedef vector[pair[vector[pair[long, vector[size_t]]], pair[long, long]]] weighted_restrictions

        ManyBodyOperator()
        ManyBodyOperator(const ManyBodyOperator&)
        ManyBodyOperator& operator=(const ManyBodyOperator&)
        ManyBodyOperator(const vector[value_type]&)
        ManyBodyOperator(const vector[key_type]&, const vector[mapped_type]&)

        ManyBodyState.mapped_type& operator[](const key_type&)
        ManyBodyState.mapped_type& at(const key_type&)
        ManyBodyState operator()(const ManyBodyState&, double)
        ManyBodyState build_restriction_mask(const restrictions&)
        void build_weighted_restriction_mask(const weighted_restrictions&)
        ManyBodyState apply(const ManyBodyState&, double)
        vector[ManyBodyState] apply(const vector[const ManyBodyState*]&, double)




        ManyBodyOperator operator-()
        ManyBodyOperator operator+(const ManyBodyOperator&, const ManyBodyOperator&)
        ManyBodyOperator operator-(const ManyBodyOperator&, const ManyBodyOperator&)
        ManyBodyOperator operator*(const ManyBodyOperator&, mapped_type)
        ManyBodyOperator operator*(mapped_type, const ManyBodyOperator&)
        ManyBodyOperator operator/(const ManyBodyOperator&, mapped_type)

        bint empty()
        size_type size()
        size_type max_size()
        void clear()
        void set_normal_ordering(bint)
        bint normal_ordering()
        size_type num_flat_terms()

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
