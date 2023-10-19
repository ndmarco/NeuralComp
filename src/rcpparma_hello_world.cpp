// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include <cmath>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
arma::mat rcpparma_hello_world() {
    arma::mat m1 = arma::eye<arma::mat>(3, 3);
    arma::mat m2 = arma::eye<arma::mat>(3, 3);
	                     
    return m1 + 3 * (m1 + m2);
}


// another simple example: outer product of a vector, 
// returning a matrix
//
// [[Rcpp::export]]
arma::mat rcpparma_outerproduct(const arma::colvec & x) {
    arma::mat m = x * x.t();
    return m;
}

// and the inner product returns a scalar
//
// [[Rcpp::export]]
double rcpparma_innerproduct(const arma::colvec & x) {
    double v = arma::as_scalar(x.t() * x);
    return v;
}


// and we can use Rcpp::List to return both at the same time
//
// [[Rcpp::export]]
Rcpp::List rcpparma_bothproducts(const arma::colvec & x) {
    arma::mat op = x * x.t();
    double    ip = arma::as_scalar(x.t() * x);
    return Rcpp::List::create(Rcpp::Named("outer")=op,
                              Rcpp::Named("inner")=ip);
}



// Returns log pdf of inverse Gaussian distribution
// [[Rcpp::export]]
double dinv_gauss(const double x,
                  const double& mean,
                  const double& shape){
  double lpdf = -1 * INFINITY;
  if(x > 0){
    lpdf = 0.5 * std::log(shape) - 0.5 * std::log(arma::datum::pi * 2 * pow(x, 3.0)) - 
      ((shape * pow(x - mean, 2.0)) / (2 * pow(mean, 2.0) * x));
  }
  
  return lpdf;
}

// [[Rcpp::export]]
double pinv_gauss(const double x,
                  const double& mean,
                  const double& shape){
  double lcdf = 0;
  if(x > 0){
    lcdf = R::pnorm((sqrt(shape / x) * ((x/ mean) - 1)), 0, 1, true, false) + (std::exp((2 * shape) / mean) 
                                                                                 * (R::pnorm(-(sqrt(shape / x) * ((x/ mean) + 1)), 0, 1, true, false)));
    lcdf = std::log(1 -lcdf);
  }
  
  return lcdf;
}


// [[Rcpp::export]]
double dnorm_test(arma::vec& prop_momentum){
  return arma::accu(arma::normpdf(prop_momentum));
}



