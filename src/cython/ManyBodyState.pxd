# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t, uint16_t, uint64_t
cimport cython

cdef extern from "ManyBodyState.cpp":
    pass

cdef extern from "ManyBodyState.h":
    cdef cppclass ManyBodyState:

        ctypedef vector[uint64_t] Key
        # ctypedef vector[uint8_t] Key
        ctypedef vector[Key] Keys
        ctypedef vector[cython.doublecomplex] Values 

        cppclass KeyComparer:
            bint operator()(const Key&, const Key&)
        cppclass ValueComparer:
            bint operator()(const cython.doublecomplex&, const cython.doublecomplex&)
        cppclass iterator:
            iterator() 
            iterator(pair[Keys.iterator, Values.iterator])
            iterator operator++()
            # iterator operator+=(int)
            iterator operator--()
            # iterator operator-=(int)
            iterator operator+(int)
            iterator operator-(int)
            int operator-(iterator)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)
            pair[Key , cython.doublecomplex] operator*()
        cppclass const_iterator:
            const_iterator()
            const_iterator(pair[Keys.const_iterator, Values.const_iterator])
            const_iterator operator++()
            # const_iterator operator+=(int)
            const_iterator operator--()
            # const_iterator operator-=(int)
            const_iterator operator+(int)
            const_iterator operator-(int)
            int operator-(const_iterator)
            bint operator==(const_iterator)
            bint operator!=(const_iterator)
            bint operator<(const_iterator)
            bint operator>(const_iterator)
            bint operator<=(const_iterator)
            bint operator>=(const_iterator)
            pair[const Key , const cython.doublecomplex] operator*()
        cppclass reverse_iterator:
            reverse_iterator()
            reverse_iterator(pair[Keys.reverse_iterator, Values.reverse_iterator])
            reverse_iterator operator++()
            # reverse_iterator operator+=(int)
            reverse_iterator operator--()
            # reverse_iterator operator-=(int)
            reverse_iterator operator+(int)
            reverse_iterator operator-(int)
            int operator-(reverse_iterator)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
            pair[Key , cython.doublecomplex] operator*()
        cppclass const_reverse_iterator:
            const_reverse_iterator()
            const_reverse_iterator(pair[Keys.const_reverse_iterator, Values.const_reverse_iterator])
            const_reverse_iterator operator++()
            # const_reverse_iterator operator+=(int)
            const_reverse_iterator operator--()
            # const_reverse_iterator operator-=(int)
            const_reverse_iterator operator+(int)
            const_reverse_iterator operator-(int)
            int operator-(const_reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(const_reverse_iterator)
            pair[const Key , const cython.doublecomplex] operator*()
        ctypedef Keys.size_type size_type 
        ctypedef pair[const Key, cython.doublecomplex] value_type;
        ctypedef Key key_type;

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
        ManyBodyState operator+(const ManyBodyState&) nogil
        ManyBodyState operator-(const ManyBodyState&) nogil
        ManyBodyState operator-() nogil
        ManyBodyState operator*(cython.doublecomplex) nogil
        ManyBodyState operator/(cython.doublecomplex) nogil
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
        iterator erase(const key_type&) nogil

        void swap(ManyBodyState&) nogil

        iterator find[K](const K&) nogil

        iterator lower_bound[K](const K&) nogil

        iterator upper_bound[K](const K&) nogil

    cdef cython.doublecomplex inner(const ManyBodyState&, const ManyBodyState&) nogil
    cdef ManyBodyState operator*(cython.doublecomplex, const ManyBodyState&) nogil
