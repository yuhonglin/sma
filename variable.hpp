// Define the variable class
// All variables are assumed as a column-major double matrices.
// Variables MUST have a name (just as in a math formula)

#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <string>
#include <iostream>
#include <exception>

#include <cstdlib>
#include <cmath>

#include "matrix.hpp"

class Variable
{
public:
  Variable(int m, int n, std::string name)
    : m_(m), n_(n),
      data_(new double[m*n]),
      lb_(new double[m*n]),
      ub_(new double[m*n]),
      btype_(new int[m*n]),
      name_(name) {
	
    for(int i = 0; i < m*n; i++) {
      lb_[i]    = -1e16;
      ub_[i]    = 1e16;
      btype_[i] = 0;  // unbounded
    }
	
  };

  Variable(const Variable& v)
    : m_(v.m()), n_(v.n()),
      data_(new double[v.m()*v.n()]),
      lb_(new double[v.m()*v.n()]),
      ub_(new double[v.m()*v.n()]),
      btype_(new int[v.m()*v.n()]),
      name_(v.name()) {

    for(int i = 0; i < m_*n_; i++) {
      lb_[i]    = v.lb()[i];
      ub_[i]    = v.ub()[i];
      btype_[i] = v.btype()[i];
    }
    
  }
  
  virtual ~Variable() = default;

  double* data() const { return data_.get(); }

  std::string name() const { return name_; }

  int m() const { return m_; }

  int n() const { return n_; }

  double* lb() const { return lb_.get(); }

  double* ub() const { return ub_.get(); }

  int* btype() const { return btype_.get(); }

  // random init with uniform distribution
  void init_unif(double l, double u, int digit = 3) {
    int base = std::pow(10, 3);
    for(int i = 0; i < m_*n_; i++) {
      data_[i] = std::rand() % base / static_cast<double>(base) * (u-l) + l;
    }
  }
    
private:
  int m_; // nrow
  int n_; // ncol
  std::unique_ptr<double[]> data_;
  std::unique_ptr<double[]> lb_;
  std::unique_ptr<double[]> ub_;
  std::unique_ptr<int[]> btype_;

  std::string name_;
};

// Helper functions
Variable operator* (const Variable&a, const Variable&b) {
  if (a.n() != b.m())
    throw std::invalid_argument("uncompatible matrix in multiplication.");

  Variable ret(a.m(), b.n(), a.name() + "*" + b.name() );
  int am = a.m();
  int an = a.n();
  int bm = b.m();
  int bn = b.n();
  FORTRAN(dgemm)(&blas::N, &blas::N, &am, &bn, &an,
		 &blas::done, a.data(), &am, b.data(), &bm, &blas::dzero, ret.data(), &am);

  return ret;
}

Variable operator- (const Variable&a, const Variable&b) {
  if (a.m() != b.m() or a.n() != b.n())
    throw std::invalid_argument("uncompatible matrix in minus.");

  Variable ret(a.m(), b.n(), a.name() + "-" + b.name() );
  for (int i = 0; i < a.m()*a.n(); i++) {
    ret.data()[i] = a.data()[i] - b.data()[i];
  }
  return ret;
}



namespace std {
  std::ostream& operator<< (std::ostream& c, const Variable& v) {
    c << v.name() << " = \n";
    for (int i = 0; i < v.m(); i++) {
      for (int j = 0; j < v.n(); j++) {
	c << v.data()[j*v.m()+i] << '\t';
      }
      c << '\n';
    }
    return c;
  }
}

#endif /* VARIABLE_H */
