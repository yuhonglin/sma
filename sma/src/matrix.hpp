// Matrix related routine
#ifndef MATRIX_H
#define MATRIX_H

// interface to BLAS

#define FORTRAN(x) x##_

namespace blas {
  double done  = 1.;
  double dzero = 0.;
  int ione  = 1;
  char T = 'T';  
  char N = 'N';
};

extern "C" {
    void FORTRAN(dgemm)(char*, char*, int*, int*, int*, double* ,
			double*, int*, double*, int*, double*, double*, int*);
}

#endif /* MATRIX_H */
