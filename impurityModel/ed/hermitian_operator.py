import numpy as np
import scipy as sp

class HermitianOperator(sp.sparse.linalg.LinearOperator):
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
        # print (f"====> HermitianOperator _matvec: self.shape = {self.shape}")
        # print (f"====> HermitianOperator _matvec: v.shape = {v.shape}")
        # res = self.diagonal.reshape(v.shape)*v + self.triangular_part @ v + self.triangular_part.getH() @ v
        # print (f"====> HermitianOperator _matvec: res.shape = {res.shape}", flush = True)
        out_shape = v.shape
        v = v.reshape((v.shape[0]))
        res = np.zeros_like(v)
        res[self.diagonal_indices] = self.diagonal*v[self.diagonal_indices]

        return (res + self.triangular_part @ v + self.triangular_part.getH() @ v).reshape(out_shape)

    def _matmat(self, m):
        # [ A00  A10* ...  AN0* ] [ m00   m01   ...  m0M ]
        # [ A10  A11  ...  AN1* ] [ m10   m11   ...  m1M ]
        # [ .    .    ...  .    ] [ .     .     ...  .   ]
        # [ AN0  .    ...  ANN  ] [ mN0   mN1   ...  mNM ]
        res = np.zeros((self.shape[0], m.shape[1]), dtype = m.dtype)
        for col in range(m.shape[1]):
            res[self.diagonal_indices, col] = self.diagonal * m[self.diagonal_indices, col]
        return res + self.triangular_part @ m + self.triangular_part.getH() @ m

    def _adjoint(self):
        return self

