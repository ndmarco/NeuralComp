#ifndef NeuralComp_SplinePrior_H
#define NeuralComp_SplinePrior_H

#include <RcppArmadillo.h>
#include <cmath>

namespace NeuralComp {

// Updates mu_A and mu_B
inline void update_mu(const arma::vec& basis_coef,
                      const arma::mat& P_mat,
                      const double I_sigma_sq,
                      arma::double mu_prior_mean,
                      arma::double mu_prior_var,
                      int iter,
                      arma::vec& mu){
  double B = (1 / mu_prior_var) + (arma::dot(arma::ones(P_mat.n_rows), P_mat * arma::ones(P_mat.n_rows)) / I_sigma_sq);
  double b = (mu_prior_mean / mu_prior_var) + (arma::dot(arma::ones(P_mat.n_rows), P_mat * basis_coef) / I_sigma_sq);
  mu(iter) = R::rnorm(B * b, std::sqrt(1 / B));
   
}

inline void update_I_sigma(const arma::vec& basis_coef,
                           const arma::mat& P_mat,
                           const double mu,
                           const double alpha,
                           const double beta,
                           int iter,
                           arma::vec& I_sigma){
  double a = alpha + (P_mat.n_cols / 2);
  double b = beta + (0.5 * arma::dot((basis_coef - mu * arma::ones(P_mat.n_rows)), P_mat * (basis_coef - mu * arma::ones(P_mat.n_rows))));
  I_sigma(iter) = 1 / R::rgamma(a, 1/b);
  
}

}