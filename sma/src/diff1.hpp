// first order difference
#ifndef DIFF1_H
#define DIFF1_H

#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "matrix.hpp"
#include "term.hpp"


class Diff1 : public Term
{
public:
  Diff1(Variable* v, int colwise = 1) : var_(v), colwise_(colwise), Term() {
    vars_.clear();
    vars_.insert(v);
    
    if (colwise == 0) { // rowwise
      num_term_ = (v->n()-1)*v->m();
      diff_.reset(new double[num_term_]);
    } else {
      num_term_ = (v->m()-1)*v->n();
      diff_.reset(new double[num_term_]);
    }
  };
  virtual ~Diff1() = default;

  virtual double func() {
    double ret = 0.;
    if (colwise_ == 0) {
      throw std::invalid_argument("Diff1: Not implemented yet.");
    } else {
      int idx = 0;
      for (int j = 0; j < var_->n(); j++) {
	for (int i = 0; i < var_->m()-1; i++) {
	  diff_[idx-j] = var_->data()[idx+1] - var_->data()[idx];
	  ret += std::pow(diff_[idx-j],2);
	  idx ++;
	}
	idx ++;
      }
      return 0.5*lambda_*ret / num_term_;
    }
  }

  virtual void   inc_grad(Variable* v, double* g) {
    double rate = lambda_ / num_term_;
    if (v != var_) {
      throw std::invalid_argument("Diff1: unkown variable");
    }
    if (colwise_ == 0) {
      throw std::invalid_argument("Diff1: not implemented yet.");
    } else {
      for (int j = 0; j < v->n(); j++) {
	g[j*var_->m()] -= rate * diff_[j*(var_->m()-1)];
	for (int i = 1; i < v->m()-1; i++) {
	  g[j*var_->m()+i] -= rate * diff_[j*(var_->m()-1) + i];
	  g[j*var_->m()+i] += rate * diff_[j*(var_->m()-1) + i + 1];
	}
	g[j*var_->m()+var_->m()-1] += rate*diff_[j*(var_->m()-1) + var_->m()-1 + 1];
      }
    }
  }
  
private:
  Variable* var_;
  int colwise_;
  int num_term_;
  std::unique_ptr<double[]> diff_;
};


#endif /* DIFF1_H */
