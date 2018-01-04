// Define the f-norm term.
#ifndef FNORM_H
#define FNORM_H

#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "matrix.hpp"
#include "term.hpp"


class FNorm : public Term
{
public:
  FNorm(Variable* v) : var_(v), Term() {
    num_term_ = v->m()*v->n();
    vars_.clear();
    vars_.insert(v);
  };
  virtual ~FNorm() = default;
  
  virtual double func() {
    double ret = 0.;
    for (int i = 0; i < var_->m()*var_->n(); i++)
      ret += std::pow(var_->data()[i],2);
    
    return 0.5*lambda_*ret / num_term_;
  }

  virtual void   inc_grad(Variable* v, double* g) {
    if (v != var_) {
      throw std::invalid_argument("FNorm: unkown variable");
    }

    double rate = lambda_ / num_term_;
    for (int i = 0; i < var_->m()*var_->n(); i++)
      g[i] += rate*v->data()[i];
  }
  
private:
  int num_term_;
  Variable* var_;
};


#endif /* FNORM_H */
