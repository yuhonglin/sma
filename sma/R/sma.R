##' Train a Matrix factorization model with specific parameters,
##' and at the same time, forecast. We will factorize a data
##' matrix Y as Y = FX and fit an auto regression model without shift 
##' on X. 
##'
##' @title Train and forecast
##' @param Y A data matrix, each row corresponds to one time series.
##' @param k The lower rank to use, i.e., ncol(F) and nrow(X)
##' @param lag The time lags used in the auto regression model built on X.
##' @param lam.nF Hyperparameter for ther ||F||^2 term.
##' @param lam.nX Hyperparameter for ther ||X||^2 term.
##' @param lam.AR Hyperparameter for ther auto regression term.
##' @param lam.nT Hyperparameter for ther ||T||^2 term where T is the
##' parameter of the auto regression model.
##' @param lam.nDF Hyperparameter for ther ||DF||^2 term.
##' @param flen Prediction length.
##' @return A list of the parameters and the forecast.
##' @author Honglin Yu
mf.train.noshift <- function(Y, k,
                             lag,
                             lam.nF,
                             lam.nX,
                             lam.AR,
                             lam.nT,
                             lam.nDF,
                             flen) {
    p = .Call("forecastc", Y,
              as.integer(k),
              as.integer(lag),
              as.numeric(lam.nF),
              as.numeric(lam.nX),
              as.numeric(lam.AR),
              as.numeric(lam.nT),
              as.numeric(lam.nDF),
              as.integer(flen));

    p$forecast = p$F%*%p$X.forecast;
    
    return (p);
}

##' Train a Matrix factorization model with specific parameters,
##' and at the same time, forecast. We will factorize a data
##' matrix Y as Y = FX and fit an auto regression model with shift 
##' on X. 
##'
##' @title Train and forecast
##' @param Y A data matrix, each row corresponds to one time series.
##' @param k The lower rank to use, i.e., ncol(F) and nrow(X)
##' @param lag The time lags used in the auto regression model built on X.
##' @param lam.nF Hyperparameter for ther ||F||^2 term.
##' @param lam.nX Hyperparameter for ther ||X||^2 term.
##' @param lam.AR Hyperparameter for ther auto regression term.
##' @param lam.nT Hyperparameter for ther ||T||^2 term where T is the
##' parameter of the auto regression model.
##' @param lam.nDF Hyperparameter for ther ||DF||^2 term.
##' @param flen Prediction length.
##' @return A list of the parameters and the forecast.
##' @author Honglin Yu
mf.train.shift <- function(Y, k,
                           lag,
                           lam.nF,
                           lam.nX,
                           lam.AR,
                           lam.nT,
                           lam.nDF,
                           flen) {
    p = .Call("forecastwsc", Y,
              as.integer(k),
              as.integer(lag),
              as.numeric(lam.nF),
              as.numeric(lam.nX),
              as.numeric(lam.AR),
              as.numeric(lam.nT),
              as.numeric(lam.nDF),
              as.integer(flen));

    p$forecast = p$F%*%p$X.forecast;
    
    return (p);
}

##' Select the hyperparameters for MF method with shift
##'
##' @param Y.train The training data, each row is a time series
##' @param Y.valid The testing data
##' @param k The lower rank to do approximation.
##' @param lag The lags to use.
##' @param lam.nF The hyperparameter of ||F||^2 term.
##' @param lam.nX The hyperparameter of ||X||^2 term.
##' @param lam.AR The hyperparameter of auto regression term.
##' @param lam.nT The hyperparameter of the ||T||^2 term where T
##' is the parameter of the auto regression.
##' @param lam.nDF The hyperparameter of the ||DF||^2 term.
##' @return The best parameters.
##' @author Honglin Yu
mf.select.hyperparam.shift <- function (Y.train,
                                        Y.valid,
                                        k,
                                        lag,
                                        lam.nF,
                                        lam.nX,
                                        lam.AR,
                                        lam.nT,
                                        lam.nDF) {
    flen = as.numeric(ncol(Y.valid))
    
    card = length(k) * length(lag) * length(lam.nF) * length(lam.nX) * length(lam.AR) * length(lam.nT) * length(lam.nDF);
    
    params = vector(mode = "list", length = card)
    mses   = vector(mode = "numeric", length = card)

    idx = 1;
    for (iter.k in k) {
        for (iter.lag in lag) {
            for (lF in lam.nF) {
                for (lX in lam.nX) {
                    for (lAR in lam.AR) {
                        for (lT in lam.nT) {
                            for (lDF in lam.nDF) {
                                res = mf.train.shift(Y.train, iter.k, iter.lag, lF, lX, lAR, lT, lDF, flen)
                                p = list(k       = iter.k,
                                         lag     = iter.lag,
                                         lam.nF  = lF,
                                         lam.nX  = lX,
                                         lam.AR  = lAR,
                                         lam.nT  = lT,
                                         lam.nDF = lDF);
                                params[[idx]] <- p
                                mses[idx] = norm(Y.valid - res$forecast, "F")
                                idx = idx + 1
                            }
                        }
                    }
                }
            }
        }
    }

    return (params[[which.min(mses)]])
}

##' Select the hyperparameters for MF method without shift
##'
##' @param Y.train The training data, each row is a time series
##' @param Y.valid The testing data
##' @param k The lower rank to do approximation.
##' @param lag The lags to use.
##' @param lam.nF The hyperparameter of ||F||^2 term.
##' @param lam.nX The hyperparameter of ||X||^2 term.
##' @param lam.AR The hyperparameter of auto regression term.
##' @param lam.nT The hyperparameter of the ||T||^2 term where T
##' is the parameter of the auto regression.
##' @param lam.nDF The hyperparameter of the ||DF||^2 term.
##' @return The best parameters.
##' @author Honglin Yu
mf.select.hyperparam.noshift <- function (Y.train,
                                          Y.valid,
                                          k,
                                          lag,
                                          lam.nF,
                                          lam.nX,
                                          lam.AR,
                                          lam.nT,
                                          lam.nDF) {
    flen = as.numeric(ncol(Y.valid))
    
    card = length(k) * length(lag) * length(lam.nF) * length(lam.nX) * length(lam.AR) * length(lam.nT) * length(lam.nDF);
    
    params = vector(mode = "list", length = card)
    mses   = vector(mode = "numeric", length = card)

    idx = 1;
    for (iter.k in k) {
        for (iter.lag in lag) {
            for (lF in lam.nF) {
                for (lX in lam.nX) {
                    for (lAR in lam.AR) {
                        for (lT in lam.nT) {
                            for (lDF in lam.nDF) {
                                res = mf.train.noshift(Y.train, iter.k, iter.lag, lF, lX, lAR, lT, lDF, flen)
                                p = list(k       = iter.k,
                                         lag     = iter.lag,
                                         lam.nF  = lF,
                                         lam.nX  = lX,
                                         lam.AR  = lAR,
                                         lam.nT  = lT,
                                         lam.nDF = lDF);
                                params[[idx]] <- p
                                mses[idx] = norm(Y.valid - res$forecast, "F")
                                idx = idx + 1
                            }
                        }
                    }
                }
            }
        }
    }

    return (params[[which.min(mses)]])
}
