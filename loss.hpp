// Define the loss function class.
// a loss function contains pointers to some terms.
// it also contains the hyperparameters of each term.

#ifndef LOSS_H
#define LOSS_H

#include <set>
#include <map>

#include "variable.hpp"
#include "term.hpp"

class Loss
{
public:
    Loss();
    virtual ~Loss();

    double func(Variable* v) {
	double ret = 0.0;
	for (auto& i : varnm_terms_[v->name()]) {
	    ret += i->func();
	}
	return ret;
    }

    // get a full objective
    double func() {
	double ret = 0.0;
	for (auto& i : terms_) {
	    ret += i->func();
	}
	return ret;
    }

    void grad(Variable* v, double* g) {
	// first clear the gradient.
	for (int i = 0; i < v->m()*v->n(); i++) {
	    g[i] = 0;
	}
	// then increment the gradient
	for (auto& i : varnm_terms_[v->name()]) {
	    i->inc_grad(v, g);
	}
    }

    void add_term(Term* t) {
	terms_.insert(t);
	vars_.insert(t->vars().begin(), t->vars().end());

	for (const auto &i : t.vars()) {
	    if (varnm_terms_.find(i->name()) == varnm_terms_.end()) {
		varnm_terms_[i->name()] = { t };
	    } else {
		varnm_terms_[i->name()].insert(t);
	    }
	}
    }

    const std::set<Variable*>& vars() { return vars_; }
    
private:
    std::map<std::string, std::set<Terms*>>    varnm_terms_;
    std::set<Variable*>                        vars_ ;
    std::set<Term*>                            terms_;
};


#endif /* LOSS_H */
