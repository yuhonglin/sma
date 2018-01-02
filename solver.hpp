#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>
#include <vector>
#include <memory>
#include <iostream>

#include "program.h"

#include "loss.hpp"
#include "variable.hpp"

class SubSolver : public Program
{
public:
  SubSolver(Loss* l, Variable* v)
    : loss_(l), var_(v),
      Program ( v->m()*v->n(),
		v->data(),
		v->lb(),
		v->ub(),
		v->btype(),
		5, 1000, 1e7, 1e-5) {};
  
  SubSolver(const SubSolver&) = delete;
  SubSolver& operator=(const SubSolver&) = delete;
  
  virtual ~SubSolver() {
  }

  double computeObjective (int n, double* x) {
    return loss_->func(var_);
  }

  void computeGradient (int n, double* x, double* g) {
    loss_->grad(var_, g);
  }
    
private:    
  Variable* var_ ;
  Loss*     loss_;
};


class Solver
{
public:
  Solver(Loss* l) : loss_(l), ftol_(1e-5), maxiter_(1000), subsolvers_(l->vars().size()) {
    int i = 0;
    for (auto &v : l->vars()) {
      subsolvers_[i].reset(new SubSolver(l, v));
      i ++;
    }
  };
  virtual ~Solver() = default;

  void solve() {
	
    int loop_idx = 0;
    double prevobj = loss_->func();
    double currobj = 0.;
    while (loop_idx <= maxiter_) {
      
      for (int i = 0; i < subsolvers_.size(); i++) {
	subsolvers_[i]->runSolver();
      }

      currobj = loss_->func();

      if (std::abs(prevobj - currobj) < prevobj*ftol_)
	break;
	    
      prevobj = currobj;
      
      loop_idx ++;
    }
  }

private:
  Solver(const Solver&);
  Solver& operator=(const Solver&);
  
  Loss * loss_;
  std::vector<std::unique_ptr<SubSolver>> subsolvers_;
  int maxiter_;
  double ftol_;
};


#endif /* SOLVER_H */
