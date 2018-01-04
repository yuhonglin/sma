train.mf.noshift <- function(Y, k, lag, lam.nF, lam.nX, lam.AR, lam.nT, lam.nDF, flen) {
    p = .Call("forecastc", Y, as.integer(k), as.integer(lag), as.numeric(lam.nF), as.numeric(lam.nX), as.numeric(lam.AR), as.numeric(lam.nT), as.numeric(lam.nDF), as.integer(flen));

    p$forecast = p$F%*%p$X.forecast;
    
    return (p);
}


train.mf.shift <- function(Y, k, lag, lam.nF, lam.nX, lam.AR, lam.nT, lam.nDF, flen) {
    p = .Call("forecastwsc", Y, as.integer(k), as.integer(lag), as.numeric(lam.nF), as.numeric(lam.nX), as.numeric(lam.AR), as.numeric(lam.nT), as.numeric(lam.nDF), as.integer(flen));

    p$forecast = p$F%*%p$X.forecast;
    
    return (p);
}


select.hyperparam.mf.shift <- function (Y.train, Y.valid, k, lag, lam.nF, lam.nX, lam.AR, lam.nT, lam.nDF) {
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
                                res = train.mf.shift(Y.train, iter.k, iter.lag, lF, lX, lAR, lT, lDF, flen)
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
