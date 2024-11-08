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

inline void update_I_sigma_cauchy(const arma::vec basis_coef,
                                  int iter,
                                  const double nu,
                                  arma::vec& omega,
                                  arma::vec& I_sigma){
  double a =  ((nu + basis_coef.n_elem) / 2);
  double b = (0.5 * arma::dot(basis_coef, basis_coef)) + (nu / omega(iter));
  I_sigma(iter) = 1 / R::rgamma(a, 1/b);
}

inline void update_omega(const double nu,
                         const double gamma,
                         int iter,
                         arma::vec& omega,
                         arma::vec& I_sigma){
  double a = (nu + 1) / 2;
  double b = (nu / I_sigma(iter)) + (1 / (gamma * gamma));
  omega(iter) = 1 / R::rgamma(a, 1/b);
}


inline double dcauchy(double x,
                      double nu,
                      double gamma){
  double pdf = std::log(1 / (2 * std::sqrt(x))) - ((nu + 1) / 2) * std::log(1 + (x / (nu * gamma * gamma)));
  return pdf;
}

inline double posterior_cauchy(double x,
                               const arma::vec basis_coef,
                               const double nu,
                               const double gamma){
  double p = dcauchy(x, nu, gamma)  + (((basis_coef.n_elem) * 0.5 *  std::log(1/x)) - (arma::dot(basis_coef, basis_coef) / (2 * x)));
  return p;
}

inline double acceptance_prob(const arma::vec basis_coef,
                              const double nu,
                              const double gamma,
                              const double prop_std,
                              double proposed_sigma,
                              double current_sigma){
  double acceptance_prob = posterior_cauchy(proposed_sigma, basis_coef, nu, gamma) - posterior_cauchy(current_sigma, basis_coef, nu, gamma) +
    R::dlnorm(current_sigma, std::log(proposed_sigma), prop_std, true) - R::dlnorm(proposed_sigma, std::log(current_sigma), prop_std, true);
  return acceptance_prob;
}

inline void RWM_cauchy(const arma::vec basis_coef,
                       int iter,
                       const double nu,
                       const double gamma,
                       const double prop_std,
                       arma::vec& I_sigma){
  double current_I_sigma = I_sigma(iter);
  double proposed_I_sigma = R::rlnorm(std::log(current_I_sigma), prop_std);
  
  if(std::log(R::runif(0,1)) < acceptance_prob(basis_coef, nu, gamma, prop_std, proposed_I_sigma, current_I_sigma)){
    I_sigma(iter) = proposed_I_sigma;
  }
}


}


#endif