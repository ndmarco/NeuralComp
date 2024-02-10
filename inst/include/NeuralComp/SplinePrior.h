#ifndef NeuralComp_SplinePrior_H
#define NeuralComp_SplinePrior_H

#include <RcppArmadillo.h>
#include <cmath>

namespace NeuralComp {

// Updates mu_A and mu_B
inline void update_mu(const arma::vec& basis_coef,
                      const double I_sigma_sq,
                      const double mu_prior_mean,
                      const double mu_prior_var,
                      int iter,
                      arma::vec& mu){
  double B = (1 / mu_prior_var) + (basis_coef.n_elem / I_sigma_sq);
  double b = (mu_prior_mean / mu_prior_var) + (arma::dot(arma::ones(basis_coef.n_elem), basis_coef) / I_sigma_sq);
  mu(iter) = R::rnorm(b / B, std::sqrt(1 / B));
   
}

inline void update_I_sigma(const arma::vec& basis_coef,
                           const double mu,
                           const double alpha,
                           const double beta,
                           int iter,
                           arma::vec& I_sigma){
  double a = alpha + ((basis_coef.n_elem - 1) / 2);
  double b = beta + (0.5 * arma::dot((basis_coef - mu * arma::ones(basis_coef.n_elem)), (basis_coef - mu * arma::ones(basis_coef.n_elem))));
  I_sigma(iter) = 1 / R::rgamma(a, 1/b);
  
}

inline void update_I_sigma_cauchy(const arma::vec& basis_coef,
                                  int iter,
                                  arma::vec& omega,
                                  arma::vec& I_sigma){
  double a = 1 + ((basis_coef.n_elem) / 2);
  double b = (0.5 * arma::dot(basis_coef, basis_coef)) + (1 / omega(iter));
  I_sigma(iter) = 1 / R::rgamma(a, 1/b);
}

inline void update_omega(const double gamma,
                         int iter,
                         arma::vec& omega,
                         arma::vec& I_sigma){
  
  double b = (1 / I_sigma(iter)) + (1 / (gamma * gamma));
  omega(iter) = 1 / R::rgamma(1, 1/b);
}


}


#endif