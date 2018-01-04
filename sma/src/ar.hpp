// Define auto regression term.
// Currently, only support basic mode: create an ar model for each row

#ifndef AR_H
#define AR_H

#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#include "matrix.hpp"
#include "term.hpp"


class AutoReg : public Term
{
public:
  AutoReg(Variable* v, Variable* p, const std::vector<int>& ls)
    : var_(v), param_(p), lags_(ls)
  {
    if (v->m() != p->m() or p->n()!=ls.size()) {
      throw std::invalid_argument("AR: incompatible inputs");
    }

    max_lag_ = *std::max_element(lags_.begin(), lags_.end());

    num_term_ = (var_->n() - max_lag_) * var_->m();
    
    diff_.reset(new double[num_term_]);

    vars_.insert(v);
    vars_.insert(p);
  };
  
  virtual ~AutoReg() = default;
  
  virtual double func() {
    double ret = 0.;
    int base = var_->m()*max_lag_;
    int diffiter = 0;
    for (int j = 0; j < var_->n()-max_lag_; j++) {
      for (int i = 0; i < var_->m(); i++) {
	diff_[diffiter] = var_->data()[base+diffiter];

	for (int k = 0; k < lags_.size(); k++) {
	  diff_[diffiter] -= param_->data()[k*param_->m()+i]
	    * var_->data()[(max_lag_+j-lags_[k])*var_->m()+i];
	}

	ret += std::pow(diff_[diffiter], 2);
	
	diffiter ++;
      }
    }

    return 0.5*lambda_*ret / num_term_;
  }

  virtual void   inc_grad(Variable* v, double* g) {
    double rate = lambda_ / num_term_;
    
    if (v == var_) {
      int base = var_->m()*max_lag_;
      int diffiter = 0;      
      for (int j = 0; j < var_->n()-max_lag_; j++) {
	for (int i = 0; i < var_->m(); i++) {
	  int vi = base+j*var_->m()+i;
	  g[vi] += rate * diff_[diffiter];
	  for(int k = 0; k < lags_.size(); k++) {
	    g[vi - lags_[k]*var_->m()] -= rate * diff_[diffiter] * param_->data()[k*param_->m()+i];
	  }
	  diffiter ++;
	}
      }
    } else if (v == param_) {
      int base = var_->m()*max_lag_;
      int diffiter = 0;      
      for (int j = 0; j < var_->n()-max_lag_; j++) {
	for (int i = 0; i < var_->m(); i++) {
	  for(int k = 0; k < lags_.size(); k++) {
	    int pi = k*param_->m() + i;
	    g[pi] -= rate * diff_[diffiter]
	      * var_->data()[(max_lag_+j-lags_[k])*var_->m() + i];
	  }
	  diffiter ++;
	}
      }
    } else {
      throw std::invalid_argument("AR: unknown variable");
    }
  }

  std::shared_ptr<Variable> forecast(int l) {
    auto ret = std::make_shared<Variable> (var_->m(), l, var_->name() + "_forcast");

#define SMA_AR_GET_VALUE(A, B)						\
    ( (B) < 0 ? (var_->data()[var_->m()*(var_->n()+(B)) + (A)])		\
      : (ret->data()[ret->m()*(B) + (A)]) )				\

    for (int i = 0; i < ret->m(); i++) {
      for (int j = 0; j < ret->n(); j++) {
	ret->data()[j*ret->m() + i] = 0;
	for (int k = 0; k < lags_.size(); k++) {
	  ret->data()[j*ret->m() + i] +=
	    SMA_AR_GET_VALUE(i, j-lags_[k]) * param_->data()[k*param_->m()+i];
	}
      }
    }

#undef SMA_AR_GET_VALUE
    
    return ret;
  }
  
public:
  Variable* param_;   // the parameter of ar model
  Variable* var_;     // the data to be build a model on
  std::vector<int> lags_;
  int max_lag_;
  int num_term_;      // number of terms in total, used in normalization
  std::unique_ptr<double[]> diff_;
};


#endif /* AR_H */
