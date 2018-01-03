train.mf <- function(Y, k, lag, lam.nF, lam.nX, lam.AR, lam.nT, lam.nDF, flen) {
    p = .Call("forecastc", Y, k, lag, lam.nF, lam.nX, lam.AR, lam.nT, lam.nDF, flen);

    p$forecast = p$F%*%p$X.forecast;
    
    return (p);
}
