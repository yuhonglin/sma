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
    vars_.clear();
    vars_.insert(v);
  };
  virtual ~FNorm() = default;
  
  virtual double func() {
    double ret = 0.;
    for (int i = 0; i < var_->m()*var_->n(); i++)
      ret += std::pow(var_->data()[i],2);
    
    return 0.5*lambda_*ret;
  }

  virtual void   inc_grad(Variable* v, double* g) {
    if (v != var_) {
      return;
    }

    for (int i = 0; i < var_->m()*var_->n(); i++)
      g[i] += lambda_*v->data()[i];
  }
  
private:
  Variable* var_;
};


#endif /* FNORM_H */
