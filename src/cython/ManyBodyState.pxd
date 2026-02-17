# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t, uint16_t, uint64_t
cimport cython

cdef extern from "ManyBodyState.cpp" nogil:
    pass

cdef extern from "ManyBodyState.h" nogil:
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

        ManyBodyState()
        ManyBodyState(const ManyBodyState&)
        ManyBodyState& operator=(const ManyBodyState&)
        ManyBodyState(const Keys&,
                      const Values&)

        bint empty() 
        size_type size() 
        size_type max_size() 
        void clear() 
        void prune(double) 
        double norm2() 
        double norm() 

        cython.doublecomplex& operator[](const key_type&) 
        cython.doublecomplex& at(const key_type&) 

        # ManyBodyState operator+=(const ManyBodyState&)
        # ManyBodyState operator-=(const ManyBodyState&)
        # ManyBodyState operator*=(const ManyBodyState&)
        # ManyBodyState operator/=(const ManyBodyState&)
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

        iterator lower_bound[K](const K&) 

        iterator upper_bound[K](const K&) 
        ManyBodyState operator*(cython.doublecomplex, const ManyBodyState&) 
        ManyBodyState operator+(const ManyBodyState&, const ManyBodyState&) 
        ManyBodyState operator-(const ManyBodyState&, const ManyBodyState&) 
        ManyBodyState operator*(const ManyBodyState&, cython.doublecomplex) 
        ManyBodyState operator/(const ManyBodyState&, cython.doublecomplex) 

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&) nogil
