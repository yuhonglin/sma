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
#include <cmath>
#include <stdexcept>

#include "matrix.hpp"
#include "term.hpp"

class MatFact : public Term
{
public:
    MatFact(int m, int n, int k, bool* mask=nullptr) :
      m_(m), n_(n), k_(k), FXY_(new double[m*n]), Term() {
	if (mask != nullptr) {
	    missing_  = true;
	    mask_     = mask;
	} else {
	    missing_  = false;
	    mask_     = nullptr;
	}
	num_term_ = m*n;
    };
    virtual ~MatFact() = default;

  virtual double func() {
	double ret = 0.;
	// first update FXY_ to F_ * X_
	FORTRAN(dgemm)(&blas::N, &blas::N, &m_, &n_, &k_,
		       &blas::done, F_->data(), &m_, X_->data(), &k_, &blas::dzero, FXY_.get(), &m_);

	// sum of squares
	if (missing_) {
	  for (int i = 0 ; i < m_*n_; i++) {
	    if (mask_[i]) {
	      FXY_[i] = FXY_[i] - Y_->data()[i];
	      ret += std::pow(FXY_[i],2);
	    }
	  }
	} else {
	  for (int i = 0 ; i < m_*n_; i++) {
	    FXY_[i] = FXY_[i] - Y_->data()[i];	    
	    ret += std::pow(FXY_[i],2);
	  }
	}

	ret *= .5;
	
	return lambda_*ret / num_term_;
    };
    
  virtual void   inc_grad(Variable* v, double* g) {
    double rate = lambda_ / num_term_;
    if (v->name() == "F") {
      // dF is stacked column by column
      //   g   = lambda* (  FX   -   Y  )  * ( X )' + g
      // (mxk)             (mxn)   (mxn)     (kxn)
      FORTRAN(dgemm)(&blas::N, &blas::T, &m_, &k_, &n_,
		     &rate, FXY_.get(), &m_, X_->data(), &k_, &blas::done, g, &m_);
    } else if (v->name() == "X") {
      // dX is stacked column by column
      //   g   =   lambda* ( F )' * (  FX   -   Y  ) + g
      // (kxn)             (mxk)      (mxn)   (mxn) 
      FORTRAN(dgemm)(&blas::T, &blas::N, &k_, &n_, &m_,
		     &rate, F_->data(), &m_, FXY_.get(), &m_, &blas::done, g, &k_);
    } else {
      throw std::invalid_argument("MatFact: unknown variable");
    }
  }

  virtual void   add_var(Variable* v) {
    if (v->name() == "F") {
      F_ = v;
      if ( v->m() != m_ or v->n() != k_) {
	throw std::invalid_argument("MatFact: invalid F");
      }
    } else if (v->name() == "X") {
      X_ = v;
      if ( v->m() != k_ or v->n() != n_) {
	throw std::invalid_argument("MatFact: invalid X");
      }
    } else {
      throw std::invalid_argument("MatFact: unknown variable");
    }
    vars_.insert(v);
  }

  virtual void  add_param(Variable* v) {
    if (v->name() == "Y") {
      Y_ = v;
      if ( v->m() != m_ or v->n() != n_) {
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

  int num_term_;
  
  std::unique_ptr<double[]> FXY_; // FX - Y

  // observers to the varialbes
  Variable* Y_;
  Variable* F_;
  Variable* X_;

  // handle missing values.
  bool  missing_;
  bool* mask_; // m x n, column major
    
};


#endif /* MF_H */
