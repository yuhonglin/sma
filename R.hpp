#include <vector>
#include <cstdlib>

#include <R.h>
#inlcude <Rinternals.h>

#include "loss.hpp"
#include "mf.hpp"
#include "fnorm.hpp"
#include "solver.hpp"
#include "ar.hpp"
#include "diff1.hpp"

extern "C" SEXP forecastc(SEXP Y_R,             // data matrix
			  SEXP k_R,             // the lower rank
			  SEXP lag_R,           // a vector of lags
			  SEXP lam_nF_R,      // lambda for ||F||^2
			  SEXP lam_nX_R,      // lambda for ||X||^2
			  SEXP lam_AR_R,      // lambda for auto regression
			  SEXP lam_nT_R,      // lambda for ||Theta||^2
			  SEXP lam_nDF_R) {   // lambda for ||DF||^2
  
  /// adapt input
  int m = Rf_nrows(y_R);
  int n = Rf_ncols(y_R);
  Variable Y(m, n, "Y");
  for (int i = 0; i < m*n; i++)
    Y.data()[i] = REAL(Y_R)[i];
  
  std::vector<int> lag;
  for (int i = 0; i < Rf_length(lag_R); i++ ) {
    lag.push_back(static_cast<int>(REAL(lag_R)[i]));
  }
  
  int k = static_cast<int>(*REAL(k));
  
  double lam_nF  = *REAL(lam_nF_R);
  double lam_nX  = *REAL(lam_nX_R);
  double lam_AR  = *REAL(lam_AR_R);
  double lam_nT  = *REAL(lam_nT_R);
  double lam_nDF = *REAL(lam_nDF_R);
  

  /// construct the model
  Variable F(m, k, "F");
  Variable X(k, n, "X");
  Variable T(k, lag.size());

  MatFact mf(m, n, k);  
  mf.add_var(&F);
  mf.add_var(&X);
  mf.add_param(&Y);

  AutoReg ar(&X, &T, lag);
  ar.set_lambda(lam_AR);
  
  FNorm nF(&F);
  nF.set_lambda(lam_nF);

  FNorm nX(&X);
  nX.set_lambda(lam_nX);

  FNorm nT(&T);
  nT.set_lambda(lam_nT);

  Diff1 nDF(&F);
  nDF.set_lambda(lam_nDF);

  Loss l;
  l.add_term(&mf);
  l.add_term(&nF);
  l.add_term(&nX);
  l.add_term(&ar);
  l.add_term(&nT);
  l.add_term(&nDF);

  // solve the model
  Solver solver(&l);
  solver.solve();

  // return value: a list of matrices: F, X and T.
  SEXP nms = PROTECT(allocVector(STRSXP, 3));
  SEXP ret = PROTECT(allocVector(VECSXP, 3));
  
  SET_STRING_ELT(nms, 0, Rf_mkCharLen("F",1));
  SEXP rF = PROTECT(Rf_allocMatrix(REALSXP, m, k));
  std::memcpy(REAL(rF), F.data(), sizeof(double) * m*k);
  SET_VECTOR_ELT(ret, 0, rF);
  UNPROTECT(1);

  SET_STRING_ELT(nms, 1, Rf_mkCharLen("X",1));
  SEXP rX = PROTECT(Rf_allocMatrix(REALSXP, k, n));
  std::memcpy(REAL(rX), X.data(), sizeof(double) * k*n);
  SET_VECTOR_ELT(ret, 1, rX);
  UNPROTECT(1);

  SET_STRING_ELT(nms, 2, Rf_mkCharLen("T",1));
  SEXP rT = PROTECT(Rf_allocMatrix(REALSXP, k, lag.size()));
  std::memcpy(REAL(rF), T.data(), sizeof(double) * k*lag.size());
  SET_VECTOR_ELT(ret, 2, rT);
  UNPROTECT(1);

  setAttrib(ret, R_NamesSymbol, nms);
  UNPROTECT(2);

  return ret;
}
