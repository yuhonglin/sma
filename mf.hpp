// The matrix factorization term.
// This term accept three variables "Y", "F" and "X" and provide
//   - compute the value given Y, F, X, that is, 0.5*|Y-FX|^2
//   - compute the gradient of F or X given Y, F, X.
//
// Note: grad() is always assumed to be called before func().

#ifndef MF_H
#define MF_H

#include <iostream>
#include <memory>
#include <stdexcept>

#include "matrix.hpp"
#include "term.hpp"

class MatFact : public Term
{
public:
    MatFact(int m, int n, int k, bool* mask=nullptr) :
	m_(m), n_(n), k_(k), FX_(new double[m*n]) Term() {
	if (mask != nullptr) {
	    missing_  = true;
	    mask_     = mask;
	} else {
	    missing_  = false;
	    mask_     = nullptr;
	}
    };
    virtual ~MatFact() = default;

    virtual double func() {
	double ret = 0.;
	// first update FX_ = F_ * X_
	FORTRAN(dgemm)(&blas::N, &blas::N, &m_, &n_, &k_,
		       &blas::done, F, &m_, X, &k_, &blas::dzero, FXY_.get(), &m_);

	// sum of squares
	if (missing_) {
	  for (int i = 0 ; i < m_*n_; i++) {
	    if (mask_[i]) {
	      FXY_[i] = FXY_[i] - Y_[i];
	      ret += std::pow(FXY_[i],2);
	    }
	  }
	} else {
	  for (int i = 0 ; i < m_*n_; i++) {
	    FXY_[i] = FXY_[i] - Y_[i];	    
	    ret += std::pow(FXY_[i],2);
	  }
	}

	ret *= .5;
	
	return lambda*ret;
    };
    
    virtual void   inc_grad(Variable* v, double* g) {
	if (v->name() == "F") {
	    // dF is stacked column by column
	    //   g   = lambda* (  FX   -   Y  )  * ( X )' + g
	    // (mxk)             (mxn)   (mxn)     (kxn)
	    FORTRAN(dgemm)(&blas::N, &blas::T, &m_, &k_, &n_,
			   &lambda_, FXY_.get(), &m_, X, &k_, &blas::done, g, &m_);
	} else if (v->name() == "X") {
	    // dX is stacked column by column
	    //   g   =   lambda* ( F )' * (  FX   -   Y  ) + g
	    // (kxn)             (mxk)      (mxn)   (mxn) 
	    FORTRAN(dgemm)(&blas::T, &blas::N, &k_, &n_, &m_,
			   &lambda_, F_.get(), &m_, FXY_.get(), &m_, &blas::done, g, &k_);
	} else {
	    throw std::invalid_argument("MatFact: unknown variable");
	}
    }

    virtual void   add_var(std::string nm, Variable* v) {
	if (nm == "F") {
	    F_ = v;
	    if ( v.m() != m_ or v.n() != k_) {
		throw std::invalid_argument("MatFact: invalid F");
	    }
	} else if (nm == "X") {
	    X_ = v;
	    if ( v.m() != k_ or v.n() != n_) {
		throw std::invalid_argument("MatFact: invalid X");
	    }
	} else if (nm == "Y") {
	    Y_ = v;
	    if ( v.m() != m_ or v.n() != n_) {
		throw std::invalid_argument("MatFact: invalid Y");
	    }
	} else {
	    throw std::invalid_argument("MatFact: unknown variable");
	}
    }
    
private:
    int m_; // nrow of Y, F
    int n_; // ncol of Y, X
    int k_; // ncol of F, nrow of X
    
    std::unique_ptr<double[]> FX_;

    // observers to the varialbes
    Variable* Y_;
    Variable* F_;
    Variable* X_;

    // handle missing values.
    bool  missing_;
    bool* mask_; // m x n, column major
    
};


#endif /* MF_H */
