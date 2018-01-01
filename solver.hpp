#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>

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
    virtual ~SubSolver() = default;

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
    Solver(Loss* l) : loss_(l), subsolvers_(l.vars().size()),
		      ftol_(1ee-5) {
	int i = 0;
	for (auto &v : l.vars()) {
	    subsolvers_[i].reset(new SubSolver(l, v));
	    i++;
	}
    };
    virtual ~Solver() = default;

    void solve() {
	
	int i = 0;
	double prevobj = mf_->func(F_,X_);
	double currobj = 0.;
	while (i <= maxiter_) {
      
	    for (int i = 0; i < subsolvers_.size(); i++) {
		subsolvers_[i]->runSolver();
	    }

	    currobj = loss_->func();

	    if (std::abs(prevobj - currobj) < prevobj*ftol_)
		break;
	    
	    prevobj = currobj;
      
	    i ++;
	}
    }

private:
    Loss * loss_;
    vector<SubSolver*> subsolvers_;
    int maxiter_;
    double ftol_;
};


#endif /* SOLVER_H */
