// Define the variable class
// All variables are assumed as a column-major double matrices.
// Variables MUST have a name (just as in a math formula)

#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <string>

class Variable
{
public:
    Variable(int m, int n, std::string name)
	: m_(m), n_(n),
	  data_(new double[m*n]),
	  lb_(new double[m*n]),
	  ub_(new double[m*n]),
	  btype_(new double[m*n]),
	  name_(name) {
	
	for(int i = 0; i < m*n; i++) {
	    lb_[i]    = -1e16;
	    ub_[i]    = 1e16;
	    btype_[i] = 0;  // unbounded
	}
	
    };
    virtual ~Variable() = default;

    double* data() { return data_; }

    std::string name() { return name_; }

    int m() { return m_; };

    int n() { return n_; };
    
private:
    int m_; // nrow
    int n_; // ncol
    std::unique_ptr<double[]> data_;
    std::unique_ptr<double[]> lb_;
    std::unique_ptr<double[]> ub_;
    std::unique_ptr<double[]> btype_;

    std::string name_;
};


#endif /* VARIABLE_H */
