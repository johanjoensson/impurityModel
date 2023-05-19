cimport numpy as np
import numpy as np
import scipy as sp
import cython
cimport cython

# cdef extern from "<complex.h>" namespace "std" nogil:
cdef extern from "<complex.h>" nogil:
    double complex conj(double complex z)

class NewHermitianOperator(sp.sparse.linalg.LinearOperator):
    # [ A00  A10* ...  AN0* ]    [ A00  0   0   ...  0   ]   [ 0    0    ...  0    ]    [ 0    A10* ...  AN0* ]
    # [ A10  A11  ...  AN1* ]  = [ 0    A11 0   ...  0   ] + [ A10  0    ...  0    ]  + [ 0    0    ...  AN1* ] 
    # [ .    .    ...  .    ]    [ 0    0   .   ...  0   ]   [ .    .    ...  .    ]    [ .    .    ...  .    ]
    # [ AN0  .    ...  ANN  ]    [ 0    0   0   ...  ANN ]   [ AN0  AN1  ...  0    ]    [ 0    .    ...  0    ]
    #         A                =         D                 +          T                          T^+
    def __init__(self, diagonal: np.ndarray, diagonal_indices: np.ndarray, triangular_part: sp.sparse.csr_matrix):
        self.shape = triangular_part.shape
        self.diagonal = diagonal if len(diagonal == 1) else diagonal.reshape((diagonal.shape[0]))
        self.diagonal_indices = diagonal_indices
        self.triangular_part = triangular_part
        self.dtype = triangular_part.dtype
        self.nnz = 2*len(triangular_part.nonzero()[0]) + len(diagonal)


    def _matvec(self, v):
        # [ A00  A10* ...  AN0* ] [ v0 ]
        # [ A10  A11  ...  AN1* ] [ v1 ]
        # [ .    .    ...  .    ] [ .  ]
        # [ AN0  .    ...  ANN  ] [ vN ]

        out_shape = v.shape
        v = v.reshape((v.shape[0]))
        return hermitian_operator_matvec(self.diagonal,
                                         self.diagonal_indices,
                                         self.triangular_part.data,
                                         self.triangular_part.indices,
                                         self.triangular_part.indptr,
                                         v,
                                         ).reshape(out_shape)

    def _matmat(self, m):
        # [ A00  A10* ...  AN0* ] [ m00   m01   ...  m0M ]
        # [ A10  A11  ...  AN1* ] [ m10   m11   ...  m1M ]
        # [ .    .    ...  .    ] [ .     .     ...  .   ]
        # [ AN0  .    ...  ANN  ] [ mN0   mN1   ...  mNM ]
        return hermitian_operator_matmat(self.diagonal,
                                         self.diagonal_indices,
                                         self.triangular_part.data,
                                         self.triangular_part.indices,
                                         self.triangular_part.indptr,
                                         m,
                                         self.shape[0]
                                         )

    def _adjoint(self):
        return self


def hermitian_operator_matvec(const double[:] diagonal,
                              const int[:] diagonal_indices,
                              const complex[:] csr_data,
                              const int[:] csr_indices,
                              const int[:] csr_index_ptr,
                              const complex[:] dense_v
                              ):
    res = np.zeros_like(dense_v)
    cdef complex[:] res_view = res
    csr_dense_matvec(diagonal,
                     diagonal_indices,
                     csr_data,
                     csr_indices,
                     csr_index_ptr,
                     dense_v,
                     res_view)
    return res

def hermitian_operator_matmat(const double[:] diagonal,
                              const int[:] diagonal_indices,
                              const complex[:] csr_data,
                              const int[:] csr_indices,
                              const int[:] csr_index_ptr,
                              const complex[:, :] dense_m,
                              const int hermitian_operator_rows
                              ):
    res = np.zeros((hermitian_operator_rows, dense_m.shape[1]), dtype = complex)
    cdef complex[:, :] res_view = res
    csr_dense_matmat(diagonal,
                     diagonal_indices,
                     csr_data,
                     csr_indices,
                     csr_index_ptr,
                     dense_m,
                     res_view)
    return res

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_matvec(const double[:] diagonal,
                           const int[:] diagonal_indices,
                           const complex[:] csr_data,
                           const int[:] csr_indices,
                           const int[:] csr_index_ptr,
                           const complex[:] dense_v, 
                           complex[:] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t csr_cols = dense_v.shape[0]
    cdef Py_ssize_t diag_i, row, k
    cdef Py_ssize_t col, index, indices_start, indices_end
    cdef complex csr_val

    for diag_i, row in enumerate(diagonal_indices):
        row = diagonal_indices[diag_i]
        res[row] = diagonal[diag_i]*dense_v[row]

    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            k, csr_val = csr_indices[index], csr_data[index]
            res[row] += csr_val*dense_v[k]
            res[k] += conj(csr_val)*dense_v[row]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void csr_dense_matmat(const double[:] diagonal,
                           const int[:] diagonal_indices,
                           const complex[:] csr_data,
                           const int[:] csr_indices,
                           const int[:] csr_index_ptr,
                           const complex[:, :] dense_m,
                           complex[:, :] res) noexcept:
    cdef Py_ssize_t csr_rows = res.shape[0]
    cdef Py_ssize_t csr_cols = dense_m.shape[0]
    cdef Py_ssize_t dense_cols = dense_m.shape[1]
    cdef Py_ssize_t diag_i, row, k
    cdef Py_ssize_t col, index, indices_start, indices_end
    cdef complex csr_val

    for diag_i, row in enumerate(diagonal_indices):
        for col in range(dense_cols):
            res[row, col] = diagonal[diag_i]*dense_m[row, col]


    for row in range(csr_rows):
        indices_start, indices_end = csr_index_ptr[row], csr_index_ptr[row + 1]
        for index in range(indices_start, indices_end):
            k, csr_val = csr_indices[index], csr_data[index]
            for col in range(dense_cols):
                res[row, col] += csr_val*dense_m[k, col]
                res[k, col] += conj(csr_val)*dense_m[row, col]
