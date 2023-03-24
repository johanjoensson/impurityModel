#ifndef IMPMOD_INTERFACE_H
#define IMPMOD_INTERFACE_H
#include <stdlib.h>
#include <complex.h>
#include <stdbool.h>
extern void run_impmod_ed(char label[], char solver_param[], double slater[], double _Complex *hyb, double _Complex* h_dft,
         double _Complex* sig, double _Complex* sig_real, double _Complex* sig_static, double _Complex* sig_dc,
         double* iw, double* w, double _Complex* rot_spherical, size_t n_orb, size_t n_iw, size_t n_w, size_t n_rot_rows, double eim, double tau,
         int verbosity, size_t size_real, size_t size_complex);
#endif // IMPMOD_INTERFACE_H
