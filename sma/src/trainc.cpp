#include <cstring>
#include <vector>
#include <cstdlib>

#include <R.h>
#include <Rinternals.h>

#undef length

#include "loss.hpp"
#include "mf.hpp"
#include "fnorm.hpp"
#include "solver.hpp"
#include "ar.hpp"
#include "arws.hpp"
#include "diff1.hpp"

extern "C" SEXP forecastc(SEXP Y_R,           // data matrix
			  SEXP k_R,           // the lower rank
			  SEXP lag_R,         // a vector of lags
			  SEXP lam_nF_R,      // lambda for ||F||^2
			  SEXP lam_nX_R,      // lambda for ||X||^2
			  SEXP lam_AR_R,      // lambda for auto regression
			  SEXP lam_nT_R,      // lambda for ||Theta||^2
			  SEXP lam_nDF_R,     // lambda for ||DF||^2
			  SEXP flen_R) {        // forecast length
  
  /// adapt input
  int m = Rf_nrows(Y_R);
  int n = Rf_ncols(Y_R);

  Variable Y(m, n, "Y");
  for (int i = 0; i < m*n; i++)
    Y.data()[i] = REAL(Y_R)[i];
  
  std::vector<int> lag;
  for (int i = 0; i < Rf_length(lag_R); i++ ) {
    lag.push_back(INTEGER(lag_R)[i]);
  }
  
  int k = static_cast<int>(*INTEGER(k_R));

  double lam_nF  = *REAL(lam_nF_R);
  double lam_nX  = *REAL(lam_nX_R);
  double lam_AR  = *REAL(lam_AR_R);
  double lam_nT  = *REAL(lam_nT_R);
  double lam_nDF = *REAL(lam_nDF_R);

  int flen = *INTEGER(flen_R);
  
  /// construct the model
  Variable F(m, k, "F");
  Variable X(k, n, "X");
  Variable T(k, lag.size(), "T");

  F.init_unif(-1,1);
  X.init_unif(-1,1);
  T.init_unif(-1,1);  
  
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

  // forecast
  auto X_forecast = ar.forecast(flen);

  // return value: a list of matrices: F, X and T.
  SEXP nms = PROTECT(allocVector(STRSXP, 4));
  SEXP ret = PROTECT(allocVector(VECSXP, 4));
  
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
  std::memcpy(REAL(rT), T.data(), sizeof(double) * k*lag.size());
  SET_VECTOR_ELT(ret, 2, rT);
  UNPROTECT(1);

  SET_STRING_ELT(nms, 3, Rf_mkCharLen("X.forecast",10));
  SEXP rXfor = PROTECT(Rf_allocMatrix(REALSXP, X_forecast->m(), X_forecast->n()));
  std::memcpy(REAL(rXfor), X_forecast->data(),
	      sizeof(double) * X_forecast->m()*X_forecast->n());
  SET_VECTOR_ELT(ret, 3, rXfor);
  UNPROTECT(1);

  
  setAttrib(ret, R_NamesSymbol, nms);
  UNPROTECT(2);

  return ret;
}


extern "C" SEXP forecastwsc(SEXP Y_R,           // data matrix
			    SEXP k_R,           // the lower rank
			    SEXP lag_R,         // a vector of lags
			    SEXP lam_nF_R,      // lambda for ||F||^2
			    SEXP lam_nX_R,      // lambda for ||X||^2
			    SEXP lam_AR_R,      // lambda for auto regression
			    SEXP lam_nT_R,      // lambda for ||Theta||^2
			    SEXP lam_nDF_R,     // lambda for ||DF||^2
			    SEXP flen_R) {        // forecast length
  /// adapt input
  int m = Rf_nrows(Y_R);
  int n = Rf_ncols(Y_R);

  Variable Y(m, n, "Y");
  for (int i = 0; i < m*n; i++)
    Y.data()[i] = REAL(Y_R)[i];
  
  std::vector<int> lag;
  for (int i = 0; i < Rf_length(lag_R); i++ ) {
    lag.push_back(INTEGER(lag_R)[i]);
  }
  
  int k = *INTEGER(k_R);

  double lam_nF  = *REAL(lam_nF_R);
  double lam_nX  = *REAL(lam_nX_R);
  double lam_AR  = *REAL(lam_AR_R);
  double lam_nT  = *REAL(lam_nT_R);
  double lam_nDF = *REAL(lam_nDF_R);

  int flen = *INTEGER(flen_R);
  
  /// construct the model
  Variable F(m, k, "F");
  Variable X(k, n, "X");
  Variable T(k, lag.size(), "T");
  Variable S(k, 1, "S");

  F.init_unif(-1,1);
  X.init_unif(-1,1);
  T.init_unif(-1,1);
  S.init_unif(-1,1);
  
  MatFact mf(m, n, k);  
  mf.add_var(&F);
  mf.add_var(&X);
  mf.add_param(&Y);

  AutoRegWithShift ar(&X, &T, &S, lag);
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

  // forecast
  auto X_forecast = ar.forecast(flen);

  // return value: a list of matrices: F, X and T.
  SEXP nms = PROTECT(allocVector(STRSXP, 5));
  SEXP ret = PROTECT(allocVector(VECSXP, 5));
  
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
  std::memcpy(REAL(rT), T.data(), sizeof(double) * k*lag.size());
  SET_VECTOR_ELT(ret, 2, rT);
  UNPROTECT(1);

  SET_STRING_ELT(nms, 3, Rf_mkCharLen("X.forecast",10));
  SEXP rXfor = PROTECT(Rf_allocMatrix(REALSXP, X_forecast->m(), X_forecast->n()));
  std::memcpy(REAL(rXfor), X_forecast->data(),
	      sizeof(double) * X_forecast->m()*X_forecast->n());
  SET_VECTOR_ELT(ret, 3, rXfor);
  UNPROTECT(1);

  SET_STRING_ELT(nms, 4, Rf_mkCharLen("S",1));
  SEXP rS = PROTECT(Rf_allocMatrix(REALSXP, k, 1));
  std::memcpy(REAL(rS), S.data(), sizeof(double) * k);
  SET_VECTOR_ELT(ret, 4, rS);
  UNPROTECT(1);
  
  
  setAttrib(ret, R_NamesSymbol, nms);
  UNPROTECT(2);

  return ret;
}
