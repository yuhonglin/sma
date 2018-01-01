// Define a term in a loss function.
// A term contains the pointer to some variables (does not hold the memory).
// A loss function the pointer to some terms (does not hold the memory).
// A term must provide
//   - function value given all its variable.
//   - gradient of its variable given all its variable values.
#ifndef TERM_H
#define TERM_H

#include <set>
#include <string>

#include "variable.hpp"

class Term
{
public:
    Term() : lambda_(1.) {};
    virtual ~Term() = default;

    // compute the value of the term.
    virtual double func() = 0;

    // compute the gradient of the term.
    virtual void   inc_grad(Variable* v, double* g) = 0;

    // add variables
    virtual void   add_var(std::string nm, Variable* v) = 0;

    void set_lambda(double l) { lambda_ = l; }

    const std::set<Variable*>& vars() { return vars_; }
    
private:
    double lambda_;
    std::set<Variable*> vars_;
    
};

#endif /* TERM_H */
