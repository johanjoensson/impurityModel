# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t

cdef extern from "ManyBodyState.cpp":
    pass

cdef extern from "ManyBodyState.h":
    cdef cppclass ManyBodyState:
        ManyBodyState()
        ManyBodyState(const ManyBodyState&)
        ManyBodyState& operator=(const ManyBodyState&)
        ManyBodyState(const vector[vector[uint8_t]]&,
                      const vector[doublecomplex]&)

        bint empty()
        size_t size()
        size_t max_size()
        void clear()
        void prune(double)
        double norm2()
        double norm()

        double complex& operator[](const vector[uint8_t]&)
        double complex& at(const vector[uint8_t]&)

        ManyBodyState operator+(const ManyBodyState&)
        ManyBodyState operator-(const ManyBodyState&)
        ManyBodyState operator-()
        ManyBodyState operator*(double complex)
        ManyBodyState operator/(double complex)
        bint operator==(const ManyBodyState&)
        bint operator!=(const ManyBodyState&)
        cppclass iterator:
            pair[vector[uint8_t], doublecomplex] operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        iterator begin()
        iterator end()

        pair[iterator, bint] insert(const pair[vector[uint8_t], doublecomplex]&)
        iterator insert(iterator, const pair[vector[uint8_t], doublecomplex]&)
        void insert[InputIt](InputIt, InputIt)

        iterator erase(iterator)
        size_t erase(const vector[uint8_t]&)

        void swap(ManyBodyState&)

        size_t count[K](const K&)

        iterator find[K](const K&)

        iterator lower_bound[K](const K&)

        iterator upper_bound[K](const K&)

    cdef double complex inner(const ManyBodyState&, const ManyBodyState&)
    cdef ManyBodyState operator*(doublecomplex, const ManyBodyState&)
