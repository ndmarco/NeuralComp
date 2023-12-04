#ifndef NeuralComp_Posterior_H
#define NeuralComp_Posterior_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include <splines2Armadillo.h>

namespace NeuralComp {
inline double log_likelihood_TI(arma::field<arma::vec>& Labels,
                                arma::vec& theta,
                                arma::vec& basis_coef_A,
                                arma::vec& basis_coef_B,
                                const arma::field<arma::mat>& basis_funct_A,
                                const arma::field<arma::mat>& basis_funct_B,
                                const arma::field<arma::mat>& basis_funct_AB,
                                const arma::field<arma::vec>& X_A,
                                const arma::field<arma::vec>& X_B,
                                const arma::field<arma::vec>& X_AB,
                                const arma::vec& n_A,
                                const arma::vec& n_B,
                                const arma::vec& n_AB){
  double l_likelihood = 0;
  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / std::exp(arma::dot(basis_funct_A(i,0).row(j), basis_coef_A))),
                                               pow((1 / theta(0)), 2));
    }
  }
  
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / std::exp(arma::dot(basis_funct_B(i,0).row(j), basis_coef_B))),
                                                   pow((1 / theta(1)), 2));
    }
  }

  // calculate log-likelihood for AB trials
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(Labels(i,0)(j) == 0){
        // label is A
        if(j != 0){
          if(Labels(i,0)(j-1) == 0){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))),
                                                     pow((1 / theta(1)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
        }
      }
    }
  }
  return l_likelihood;
}

inline double log_likelihood_TI_A_only(arma::field<arma::vec>& Labels,
                                       arma::vec& theta,
                                       arma::vec& basis_coef_A,
                                       arma::vec& basis_coef_B,
                                       const arma::field<arma::mat>& basis_funct_A,
                                       const arma::field<arma::mat>& basis_funct_B,
                                       const arma::field<arma::mat>& basis_funct_AB,
                                       const arma::field<arma::vec>& X_A,
                                       const arma::field<arma::vec>& X_B,
                                       const arma::field<arma::vec>& X_AB,
                                       const arma::vec& n_A,
                                       const arma::vec& n_B,
                                       const arma::vec& n_AB){
  double l_likelihood = 0;
  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / std::exp(arma::dot(basis_funct_A(i,0).row(j), basis_coef_A))),
                                               pow((1 / theta(0)), 2));
    }
  }
  
  // calculate log-likelihood for AB trials
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(Labels(i,0)(j) == 0){
        // label is A
        if(j != 0){
          if(Labels(i,0)(j-1) == 0){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))),
                                                     pow((1 / theta(1)), 2)) +
                                                       dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
        }
      }
    }
  }
  return l_likelihood;
}

inline double log_likelihood_TI_B_only(arma::field<arma::vec>& Labels,
                                       arma::vec& theta,
                                       arma::vec& basis_coef_A,
                                       arma::vec& basis_coef_B,
                                       const arma::field<arma::mat>& basis_funct_A,
                                       const arma::field<arma::mat>& basis_funct_B,
                                       const arma::field<arma::mat>& basis_funct_AB,
                                       const arma::field<arma::vec>& X_A,
                                       const arma::field<arma::vec>& X_B,
                                       const arma::field<arma::vec>& X_AB,
                                       const arma::vec& n_A,
                                       const arma::vec& n_B,
                                       const arma::vec& n_AB){
  double l_likelihood = 0;
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / std::exp(arma::dot(basis_funct_B(i,0).row(j), basis_coef_B))),
                                               pow((1 / theta(1)), 2));
    }
  }
  
  // calculate log-likelihood for AB trials
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(Labels(i,0)(j) == 0){
        // label is A
        if(j != 0){
          if(Labels(i,0)(j-1) == 0){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))),
                                                     pow((1 / theta(1)), 2)) +
                                                       dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(2), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A))), pow((1 / theta(0)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B))), pow((1 / theta(1)), 2));
        }
      }
    }
  }
  return l_likelihood;
}

// transform the parameters into an unbounded space
// theta: (I_A, I_B, sigma_A, sigma_B, delta)
inline double log_likelihood(arma::field<arma::vec>& Labels,
                             arma::vec& theta,
                             const arma::field<arma::vec>& X_A,
                             const arma::field<arma::vec>& X_B,
                             const arma::field<arma::vec>& X_AB,
                             const arma::vec& n_A,
                             const arma::vec& n_B,
                             const arma::vec& n_AB){
  double l_likelihood = 0;
  
  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
    }
  }
  
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
    }
  }
  
  // calculate log-likelihood for AB trials
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(Labels(i,0)(j) == 0){
        // label is A
        if(j != 0){
          if(Labels(i,0)(j-1) == 0){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
        }
      }
    }
  }
  return l_likelihood;
}

inline double log_posterior(arma::field<arma::vec>& Labels,
                            arma::vec theta,
                            const arma::field<arma::vec>& X_A,
                            const arma::field<arma::vec>& X_B,
                            const arma::field<arma::vec>& X_AB,
                            const arma::vec& n_A,
                            const arma::vec& n_B,
                            const arma::vec& n_AB,
                            const double& I_A_shape, 
                            const double& I_A_rate,
                            const double& I_B_shape,
                            const double& I_B_rate,
                            const double& sigma_A_mean,
                            const double& sigma_A_shape,
                            const double& sigma_B_mean,
                            const double& sigma_B_shape){
  double l_posterior = log_likelihood(Labels, theta, X_A, X_B, X_AB, n_A, n_B, n_AB) +
    log_prior(I_A_shape, I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
              sigma_B_mean, sigma_B_shape, theta);
  return l_posterior;
}

inline double log_posterior_TI(arma::field<arma::vec>& Labels,
                               arma::vec theta,
                               arma::vec basis_coef_A,
                               arma::vec basis_coef_B,
                               const arma::field<arma::mat>& basis_funct_A,
                               const arma::field<arma::mat>& basis_funct_B,
                               const arma::field<arma::mat>& basis_funct_AB,
                               const arma::field<arma::vec>& X_A,
                               const arma::field<arma::vec>& X_B,
                               const arma::field<arma::vec>& X_AB,
                               const arma::vec& n_A,
                               const arma::vec& n_B,
                               const arma::vec& n_AB,
                               const double& mu_A, 
                               const double& mu_B,
                               const double& I_A_sigma_sq,
                               const double& I_B_sigma_sq,
                               const double& sigma_A_mean,
                               const double& sigma_A_shape,
                               const double& sigma_B_mean,
                               const double& sigma_B_shape,
                               const arma::mat P_mat){
  double l_posterior = log_likelihood_TI(Labels, theta, basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB) +
    log_prior_TI(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean, sigma_A_shape,
                 sigma_B_mean, sigma_B_shape, P_mat, theta, basis_coef_A, basis_coef_B);
  return l_posterior;
}


inline double log_posterior_FR(arma::field<arma::vec>& Labels,
                               arma::vec theta,
                               arma::vec basis_coef_A,
                               arma::vec basis_coef_B,
                               const arma::field<arma::mat>& basis_funct_A,
                               const arma::field<arma::mat>& basis_funct_B,
                               const arma::field<arma::mat>& basis_funct_AB,
                               const arma::field<arma::vec>& X_A,
                               const arma::field<arma::vec>& X_B,
                               const arma::field<arma::vec>& X_AB,
                               const arma::vec& n_A,
                               const arma::vec& n_B,
                               const arma::vec& n_AB,
                               const double& mu_A, 
                               const double& mu_B,
                               const double& I_A_sigma_sq,
                               const double& I_B_sigma_sq,
                               const arma::mat P_mat){
  double l_posterior = log_likelihood_TI(Labels, theta, basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                           log_prior_FR(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq,
                                                        P_mat, basis_coef_A, basis_coef_B);
  return l_posterior;
}

inline double log_posterior_FR_A_only(arma::field<arma::vec>& Labels,
                                      arma::vec theta,
                                      arma::vec basis_coef_A,
                                      arma::vec basis_coef_B,
                                      const arma::field<arma::mat>& basis_funct_A,
                                      const arma::field<arma::mat>& basis_funct_B,
                                      const arma::field<arma::mat>& basis_funct_AB,
                                      const arma::field<arma::vec>& X_A,
                                      const arma::field<arma::vec>& X_B,
                                      const arma::field<arma::vec>& X_AB,
                                      const arma::vec& n_A,
                                      const arma::vec& n_B,
                                      const arma::vec& n_AB,
                                      const double& mu_A, 
                                      const double& mu_B,
                                      const double& I_A_sigma_sq,
                                      const double& I_B_sigma_sq,
                                      const arma::mat P_mat){
  
  double l_posterior = log_likelihood_TI_A_only(Labels, theta, basis_coef_A, basis_coef_B,
                                                basis_funct_A, basis_funct_B, basis_funct_AB,
                                                X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                                  log_prior_FR(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                                                               P_mat, basis_coef_A, basis_coef_B);
  return l_posterior;
}

inline double log_posterior_FR_B_only(arma::field<arma::vec>& Labels,
                                      arma::vec theta,
                                      arma::vec basis_coef_A,
                                      arma::vec basis_coef_B,
                                      const arma::field<arma::mat>& basis_funct_A,
                                      const arma::field<arma::mat>& basis_funct_B,
                                      const arma::field<arma::mat>& basis_funct_AB,
                                      const arma::field<arma::vec>& X_A,
                                      const arma::field<arma::vec>& X_B,
                                      const arma::field<arma::vec>& X_AB,
                                      const arma::vec& n_A,
                                      const arma::vec& n_B,
                                      const arma::vec& n_AB,
                                      const double& mu_A, 
                                      const double& mu_B,
                                      const double& I_A_sigma_sq,
                                      const double& I_B_sigma_sq,
                                      const arma::mat P_mat){
  
  double l_posterior = log_likelihood_TI_B_only(Labels, theta, basis_coef_A, basis_coef_B,
                                                basis_funct_A, basis_funct_B, basis_funct_AB,
                                                X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                                  log_prior_FR(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                                                               P_mat, basis_coef_A, basis_coef_B);
  return l_posterior;
}

inline double log_posterior_sigma(arma::field<arma::vec>& Labels,
                                  arma::vec theta,
                                  arma::vec basis_coef_A,
                                  arma::vec basis_coef_B,
                                  const arma::field<arma::mat>& basis_funct_A,
                                  const arma::field<arma::mat>& basis_funct_B,
                                  const arma::field<arma::mat>& basis_funct_AB,
                                  const arma::field<arma::vec>& X_A,
                                  const arma::field<arma::vec>& X_B,
                                  const arma::field<arma::vec>& X_AB,
                                  const arma::vec& n_A,
                                  const arma::vec& n_B,
                                  const arma::vec& n_AB,
                                  const double& sigma_A_mean,
                                  const double& sigma_A_shape,
                                  const double& sigma_B_mean,
                                  const double& sigma_B_shape){
  double l_posterior = log_likelihood_TI(Labels, theta, basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                           log_prior_sigma(sigma_A_mean, sigma_A_shape,
                                                           sigma_B_mean, sigma_B_shape,
                                                           theta);
  return l_posterior;
}

inline double log_posterior_TI_A_only(arma::field<arma::vec>& Labels,
                                      arma::vec theta,
                                      arma::vec basis_coef_A,
                                      arma::vec basis_coef_B,
                                      const arma::field<arma::mat>& basis_funct_A,
                                      const arma::field<arma::mat>& basis_funct_B,
                                      const arma::field<arma::mat>& basis_funct_AB,
                                      const arma::field<arma::vec>& X_A,
                                      const arma::field<arma::vec>& X_B,
                                      const arma::field<arma::vec>& X_AB,
                                      const arma::vec& n_A,
                                      const arma::vec& n_B,
                                      const arma::vec& n_AB,
                                      const double& mu_A, 
                                      const double& mu_B,
                                      const double& I_A_sigma_sq,
                                      const double& I_B_sigma_sq,
                                      const double& sigma_A_mean,
                                      const double& sigma_A_shape,
                                      const double& sigma_B_mean,
                                      const double& sigma_B_shape,
                                      const arma::mat P_mat){

  double l_posterior = log_likelihood_TI_A_only(Labels, theta, basis_coef_A, basis_coef_B,
                                                basis_funct_A, basis_funct_B, basis_funct_AB,
                                                X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                                  log_prior_TI(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean, sigma_A_shape,
                                                        sigma_B_mean, sigma_B_shape, P_mat, theta, basis_coef_A, basis_coef_B);
  return l_posterior;
}
inline double log_posterior_TI_B_only(arma::field<arma::vec>& Labels,
                                      arma::vec theta,
                                      arma::vec basis_coef_A,
                                      arma::vec basis_coef_B,
                                      const arma::field<arma::mat>& basis_funct_A,
                                      const arma::field<arma::mat>& basis_funct_B,
                                      const arma::field<arma::mat>& basis_funct_AB,
                                      const arma::field<arma::vec>& X_A,
                                      const arma::field<arma::vec>& X_B,
                                      const arma::field<arma::vec>& X_AB,
                                      const arma::vec& n_A,
                                      const arma::vec& n_B,
                                      const arma::vec& n_AB,
                                      const double& mu_A, 
                                      const double& mu_B,
                                      const double& I_A_sigma_sq,
                                      const double& I_B_sigma_sq,
                                      const double& sigma_A_mean,
                                      const double& sigma_A_shape,
                                      const double& sigma_B_mean,
                                      const double& sigma_B_shape,
                                      const arma::mat P_mat){
  double l_posterior = log_likelihood_TI_B_only(Labels, theta, basis_coef_A, basis_coef_B,
                                                basis_funct_A, basis_funct_B, basis_funct_AB,
                                                X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                           log_prior_TI(mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean, sigma_A_shape,
                                                        sigma_B_mean, sigma_B_shape, P_mat, theta, basis_coef_A, basis_coef_B);
  return l_posterior;
}
inline double log_posterior_delta(arma::field<arma::vec>& Labels,
                                  arma::vec theta,
                                  const arma::field<arma::vec>& X_A,
                                  const arma::field<arma::vec>& X_B,
                                  const arma::field<arma::vec>& X_AB,
                                  const arma::vec& n_A,
                                  const arma::vec& n_B,
                                  const arma::vec& n_AB,
                                  const double& delta_shape,
                                  const double& delta_rate){
  double l_posterior = log_likelihood(Labels, theta, X_A, X_B, X_AB, n_A, n_B, n_AB) +
    log_prior_delta(delta_shape, delta_rate, theta);
  return l_posterior;
}

inline double transformed_log_posterior_delta(arma::field<arma::vec>& Labels,
                                              arma::vec theta,
                                              const arma::field<arma::vec>& X_A,
                                              const arma::field<arma::vec>& X_B,
                                              const arma::field<arma::vec>& X_AB,
                                              const arma::vec& n_A,
                                              const arma::vec& n_B,
                                              const arma::vec& n_AB,
                                              const double& delta_shape,
                                              const double& delta_rate){
  double l_posterior = log_posterior_delta(Labels, transform_pars(theta), X_A, X_B, X_AB, n_A, n_B, n_AB,
                                           delta_shape, delta_rate) + theta(4);
  return l_posterior;
}

inline double transformed_log_posterior(arma::field<arma::vec>& Labels,
                                        arma::vec& theta,
                                        const arma::field<arma::vec>& X_A,
                                        const arma::field<arma::vec>& X_B,
                                        const arma::field<arma::vec>& X_AB,
                                        const arma::vec& n_A,
                                        const arma::vec& n_B,
                                        const arma::vec& n_AB,
                                        const double& I_A_shape, 
                                        const double& I_A_rate,
                                        const double& I_B_shape,
                                        const double& I_B_rate,
                                        const double& sigma_A_mean,
                                        const double& sigma_A_shape,
                                        const double& sigma_B_mean,
                                        const double& sigma_B_shape){
  double l_posterior = log_posterior(Labels, transform_pars(theta), X_A, X_B, X_AB,
                                     n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                     I_B_rate, sigma_A_mean, sigma_A_shape,
                                     sigma_B_mean, sigma_B_shape) + arma::accu(theta.subvec(0,3));
  return l_posterior;
}

inline double transformed_log_posterior_TI(arma::field<arma::vec>& Labels,
                                           arma::vec theta,
                                           arma::vec& basis_coef_A,
                                           arma::vec& basis_coef_B,
                                           const arma::field<arma::mat>& basis_funct_A,
                                           const arma::field<arma::mat>& basis_funct_B,
                                           const arma::field<arma::mat>& basis_funct_AB,
                                           const arma::field<arma::vec>& X_A,
                                           const arma::field<arma::vec>& X_B,
                                           const arma::field<arma::vec>& X_AB,
                                           const arma::vec& n_A,
                                           const arma::vec& n_B,
                                           const arma::vec& n_AB,
                                           const double& mu_A, 
                                           const double& mu_B,
                                           const double& I_A_sigma_sq,
                                           const double& I_B_sigma_sq,
                                           const double& sigma_A_mean,
                                           const double& sigma_A_shape,
                                           const double& sigma_B_mean,
                                           const double& sigma_B_shape,
                                           const arma::mat P_mat){
  double l_posterior = log_posterior_TI(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, 
                                         I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean, 
                                         sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) +
                                           theta(0) + theta(1);
  return l_posterior;
}

inline double transformed_log_posterior_FR(arma::field<arma::vec>& Labels,
                                           arma::vec theta,
                                           arma::vec& basis_coef_A,
                                           arma::vec& basis_coef_B,
                                           const arma::field<arma::mat>& basis_funct_A,
                                           const arma::field<arma::mat>& basis_funct_B,
                                           const arma::field<arma::mat>& basis_funct_AB,
                                           const arma::field<arma::vec>& X_A,
                                           const arma::field<arma::vec>& X_B,
                                           const arma::field<arma::vec>& X_AB,
                                           const arma::vec& n_A,
                                           const arma::vec& n_B,
                                           const arma::vec& n_AB,
                                           const double& mu_A, 
                                           const double& mu_B,
                                           const double& I_A_sigma_sq,
                                           const double& I_B_sigma_sq,
                                           const arma::mat P_mat){
  double l_posterior = log_posterior_FR(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                        basis_funct_A, basis_funct_B, basis_funct_AB,
                                        X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, 
                                        I_A_sigma_sq, I_B_sigma_sq, P_mat);
  return l_posterior;
}

inline double transformed_log_posterior_sigma(arma::field<arma::vec>& Labels,
                                              arma::vec theta,
                                              arma::vec& basis_coef_A,
                                              arma::vec& basis_coef_B,
                                              const arma::field<arma::mat>& basis_funct_A,
                                              const arma::field<arma::mat>& basis_funct_B,
                                              const arma::field<arma::mat>& basis_funct_AB,
                                              const arma::field<arma::vec>& X_A,
                                              const arma::field<arma::vec>& X_B,
                                              const arma::field<arma::vec>& X_AB,
                                              const arma::vec& n_A,
                                              const arma::vec& n_B,
                                              const arma::vec& n_AB,
                                              const double& sigma_A_mean,
                                              const double& sigma_A_shape,
                                              const double& sigma_B_mean,
                                              const double& sigma_B_shape){
  double l_posterior = log_posterior_sigma(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                        basis_funct_A, basis_funct_B, basis_funct_AB,
                                        X_A, X_B, X_AB, n_A, n_B, n_AB, sigma_A_mean, 
                                        sigma_A_shape,sigma_B_mean, sigma_B_shape) +
                                          theta(0) + theta(1);
  return l_posterior;
}


// Calculate gradient of log_posterior on the original scale
inline arma::vec calc_gradient(arma::field<arma::vec>& Labels,
                               arma::vec theta,
                               const arma::field<arma::vec>& X_A,
                               const arma::field<arma::vec>& X_B,
                               const arma::field<arma::vec>& X_AB,
                               const arma::vec& n_A,
                               const arma::vec& n_B,
                               const arma::vec& n_AB,
                               const double& I_A_shape, 
                               const double& I_A_rate,
                               const double& I_B_shape,
                               const double& I_B_rate,
                               const double& sigma_A_mean,
                               const double& sigma_A_shape,
                               const double& sigma_B_mean,
                               const double& sigma_B_shape,
                               const arma::vec& eps_step){
  arma::vec grad(theta.n_elem - 1, arma::fill::zeros);
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  for(int i = 0; i < theta.n_elem - 1; i++){
    theta_p_eps = theta;
    // f(x + e) in the i^th dimension
    theta_p_eps(i) = theta_p_eps(i) + eps_step(i);
    theta_m_eps = theta;
    // f(x - e) in the i^th dimension
    theta_m_eps(i) = theta_m_eps(i) - eps_step(i);
    // approximate gradient ((f(x + e) f(x - e))/ 2e)
    grad(i) = (log_posterior(Labels, theta_p_eps, X_A, X_B, X_AB,
               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
               I_B_rate, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape) - log_posterior(Labels, theta_m_eps,
               X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
               I_B_rate, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape)) / (2 * eps_step(i));
    
  }
  return grad;
}

inline arma::vec calc_gradient_TI(arma::field<arma::vec>& Labels,
                                  arma::vec theta,
                                  arma::vec basis_coef_A,
                                  arma::vec basis_coef_B,
                                  const arma::field<arma::mat>& basis_funct_A,
                                  const arma::field<arma::mat>& basis_funct_B,
                                  const arma::field<arma::mat>& basis_funct_AB,
                                  const arma::field<arma::vec>& X_A,
                                  const arma::field<arma::vec>& X_B,
                                  const arma::field<arma::vec>& X_AB,
                                  const arma::vec& n_A,
                                  const arma::vec& n_B,
                                  const arma::vec& n_AB,
                                  const double& mu_A, 
                                  const double& mu_B,
                                  const double& I_A_sigma_sq,
                                  const double& I_B_sigma_sq,
                                  const double& sigma_A_mean,
                                  const double& sigma_A_shape,
                                  const double& sigma_B_mean,
                                  const double& sigma_B_shape,
                                  const arma::mat P_mat,
                                  const double eps_step){
  arma::vec grad((theta.n_elem + basis_coef_A.n_elem + basis_coef_B.n_elem), arma::fill::zeros);
  arma::vec ph_p_eps = grad;
  ph_p_eps.subvec(0, basis_coef_A.n_elem -1) = basis_coef_A;
  ph_p_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1) = basis_coef_B;
  ph_p_eps.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, grad.n_elem - 1) = theta;
  arma::vec ph_m_eps = ph_p_eps;
  arma::vec ph = ph_p_eps;
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  arma::vec basis_coef_A_p_eps = basis_coef_A;
  arma::vec basis_coef_A_m_eps = basis_coef_A;
  arma::vec basis_coef_B_p_eps = basis_coef_B;
  arma::vec basis_coef_B_m_eps = basis_coef_B;

  for(int i = 0; i < grad.n_elem - 1; i++){
    ph_p_eps = ph;
    // f(x + e) in the i^th dimension
    ph_p_eps(i) = ph_p_eps(i) + eps_step;
    theta_p_eps = ph_p_eps.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, grad.n_elem - 1);
    basis_coef_A_p_eps = ph_p_eps.subvec(0, basis_coef_A.n_elem - 1);
    basis_coef_B_p_eps = ph_p_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
    
    ph_m_eps = ph;
    // f(x - e) in the i^th dimension
    ph_m_eps(i) = ph_m_eps(i) - eps_step;
    theta_m_eps = ph_m_eps.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, grad.n_elem - 1);
    basis_coef_A_m_eps = ph_m_eps.subvec(0, basis_coef_A.n_elem - 1);
    basis_coef_B_m_eps = ph_m_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
    // approximate gradient ((f(x + e) f(x - e))/ 2e)
    
    if(i < basis_coef_A.n_elem){
      grad(i) = (log_posterior_TI_A_only(Labels, theta_p_eps, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) - log_posterior_TI_A_only(Labels, 
                 theta_m_eps, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat)) / (2 * eps_step);
    }else if(i < basis_coef_B.n_elem + basis_coef_A.n_elem){
      grad(i) = (log_posterior_TI_B_only(Labels, theta_p_eps, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) - log_posterior_TI_B_only(Labels, 
                 theta_m_eps, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat)) / (2 * eps_step);
    }else if(i == basis_coef_B.n_elem + basis_coef_A.n_elem){
      grad(i) = (log_posterior_TI_A_only(Labels, theta_p_eps, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) - log_posterior_TI_A_only(Labels, 
                 theta_m_eps, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat)) / (2 * eps_step);
    }else{
      grad(i) = (log_posterior_TI_B_only(Labels, theta_p_eps, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) - log_posterior_TI_B_only(Labels, 
                 theta_m_eps, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat)) / (2 * eps_step);
    }
    // grad(i) = (log_posterior_TI(Labels, theta_p_eps, basis_coef_A_p_eps, basis_coef_B_p_eps, 
    //            basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
    //            X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
    //            sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat) - log_posterior_TI(Labels, 
    //            theta_m_eps, basis_coef_A_m_eps, basis_coef_B_m_eps,
    //            basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
    //            X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
    //            sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, P_mat)) / (2 * eps_step);
  }
  return grad.subvec(0, grad.n_elem -2);
}


inline arma::vec calc_gradient_FR(arma::field<arma::vec>& Labels,
                                  arma::vec theta,
                                  arma::vec basis_coef_A,
                                  arma::vec basis_coef_B,
                                  const arma::field<arma::mat>& basis_funct_A,
                                  const arma::field<arma::mat>& basis_funct_B,
                                  const arma::field<arma::mat>& basis_funct_AB,
                                  const arma::field<arma::vec>& X_A,
                                  const arma::field<arma::vec>& X_B,
                                  const arma::field<arma::vec>& X_AB,
                                  const arma::vec& n_A,
                                  const arma::vec& n_B,
                                  const arma::vec& n_AB,
                                  const double& mu_A, 
                                  const double& mu_B,
                                  const double& I_A_sigma_sq,
                                  const double& I_B_sigma_sq,
                                  const arma::mat P_mat,
                                  const double eps_step){
  arma::vec grad((basis_coef_A.n_elem + basis_coef_B.n_elem), arma::fill::zeros);
  arma::vec ph_p_eps = grad;
  ph_p_eps.subvec(0, basis_coef_A.n_elem -1) = basis_coef_A;
  ph_p_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1) = basis_coef_B;
  arma::vec ph_m_eps = ph_p_eps;
  arma::vec ph = ph_p_eps;
  arma::vec basis_coef_A_p_eps = basis_coef_A;
  arma::vec basis_coef_A_m_eps = basis_coef_A;
  arma::vec basis_coef_B_p_eps = basis_coef_B;
  arma::vec basis_coef_B_m_eps = basis_coef_B;
  
  for(int i = 0; i < grad.n_elem; i++){
    ph_p_eps = ph;
    // f(x + e) in the i^th dimension
    ph_p_eps(i) = ph_p_eps(i) + eps_step;
    basis_coef_A_p_eps = ph_p_eps.subvec(0, basis_coef_A.n_elem - 1);
    basis_coef_B_p_eps = ph_p_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
    
    ph_m_eps = ph;
    // f(x - e) in the i^th dimension
    ph_m_eps(i) = ph_m_eps(i) - eps_step;
    basis_coef_A_m_eps = ph_m_eps.subvec(0, basis_coef_A.n_elem - 1);
    basis_coef_B_m_eps = ph_m_eps.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
    // approximate gradient ((f(x + e) f(x - e))/ 2e)
    
    if(i < basis_coef_A.n_elem){
      grad(i) = (log_posterior_FR_A_only(Labels, theta, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                P_mat) - log_posterior_FR_A_only(Labels, 
                 theta, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                P_mat)) / (2 * eps_step);
    }else{
      grad(i) = (log_posterior_FR_B_only(Labels, theta, basis_coef_A_p_eps, basis_coef_B_p_eps, 
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 P_mat) - log_posterior_FR_B_only(Labels, 
                 theta, basis_coef_A_m_eps, basis_coef_B_m_eps,
                 basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
                 X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                 P_mat)) / (2 * eps_step);
    }
  }
  return grad;
}

inline arma::vec calc_gradient_sigma(arma::field<arma::vec>& Labels,
                                     arma::vec theta,
                                     arma::vec basis_coef_A,
                                     arma::vec basis_coef_B,
                                     const arma::field<arma::mat>& basis_funct_A,
                                     const arma::field<arma::mat>& basis_funct_B,
                                     const arma::field<arma::mat>& basis_funct_AB,
                                     const arma::field<arma::vec>& X_A,
                                     const arma::field<arma::vec>& X_B,
                                     const arma::field<arma::vec>& X_AB,
                                     const arma::vec& n_A,
                                     const arma::vec& n_B,
                                     const arma::vec& n_AB,
                                     const double& sigma_A_mean,
                                     const double& sigma_A_shape,
                                     const double& sigma_B_mean,
                                     const double& sigma_B_shape,
                                     const double eps_step){
  arma::vec grad((theta.n_elem), arma::fill::zeros);
  arma::vec ph_p_eps = theta;
  arma::vec ph_m_eps = theta;
  arma::vec ph = ph_p_eps;
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  
  for(int i = 0; i < grad.n_elem - 1; i++){
    ph_p_eps = ph;
    // f(x + e) in the i^th dimension
    ph_p_eps(i) = ph_p_eps(i) + eps_step;
    theta_p_eps = ph_p_eps;
    
    ph_m_eps = ph;
    // f(x - e) in the i^th dimension
    ph_m_eps(i) = ph_m_eps(i) - eps_step;
    theta_m_eps = ph_m_eps;
    
    // approximate gradient ((f(x + e) f(x - e))/ 2e)
    grad(i) = (log_posterior_sigma(Labels, theta_p_eps, basis_coef_A, basis_coef_B, 
               basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
               X_B, X_AB, n_A, n_B, n_AB, 
               sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape) - log_posterior_sigma(Labels, 
               theta_m_eps, basis_coef_A, basis_coef_B,
               basis_funct_A, basis_funct_B, basis_funct_AB, X_A,
               X_B, X_AB, n_A, n_B, n_AB, 
               sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape)) / (2 * eps_step);
    
  }
  return grad.subvec(0, grad.n_elem -2);
}

// Calculate derivative of log_posterior with respect to delta on the original scale
inline double calc_gradient_delta(arma::field<arma::vec>& Labels,
                                  arma::vec theta,
                                  const arma::field<arma::vec>& X_A,
                                  const arma::field<arma::vec>& X_B,
                                  const arma::field<arma::vec>& X_AB,
                                  const arma::vec& n_A,
                                  const arma::vec& n_B,
                                  const arma::vec& n_AB,
                                  const double& delta_shape,
                                  const double& delta_rate,
                                  const arma::vec& eps_step){
  double deriv = 0;
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  theta_p_eps = theta;
  // f(x + e) in the i^th dimension
  theta_p_eps(4) = theta_p_eps(4) + eps_step(4);
  theta_m_eps = theta;
  // f(x - e) in the i^th dimension
  theta_m_eps(4) = theta_m_eps(4) - eps_step(4);
  // approximate gradient ((f(x + e) f(x - e))/ 2e)
  deriv = (log_posterior_delta(Labels, theta_p_eps, X_A, X_B, X_AB,
           n_A, n_B, n_AB, delta_shape, delta_rate) - 
           log_posterior_delta(Labels, theta_m_eps, X_A, X_B, X_AB,
           n_A, n_B, n_AB, delta_shape, delta_rate)) / (2 * eps_step(4));
    
  return deriv;
}

inline double trans_calc_gradient_delta(arma::field<arma::vec>& Labels,
                                        arma::vec& theta,
                                        const arma::field<arma::vec>& X_A,
                                        const arma::field<arma::vec>& X_B,
                                        const arma::field<arma::vec>& X_AB,
                                        const arma::vec& n_A,
                                        const arma::vec& n_B,
                                        const arma::vec& n_AB,
                                        const double& delta_shape,
                                        const double& delta_rate,
                                        const arma::vec& eps_step){
  
  double grad = calc_gradient_delta(Labels, transform_pars(theta), X_A, X_B, X_AB,
                                    n_A, n_B, n_AB, delta_shape, delta_rate, 
                                    eps_step) + 1;
  
  return grad;
}

// calculate gradient of log_posterior on transformed space of parameters
inline arma::vec trans_calc_gradient(arma::field<arma::vec>& Labels,
                                     arma::vec& theta,
                                     const arma::field<arma::vec>& X_A,
                                     const arma::field<arma::vec>& X_B,
                                     const arma::field<arma::vec>& X_AB,
                                     const arma::vec& n_A,
                                     const arma::vec& n_B,
                                     const arma::vec& n_AB,
                                     const double& I_A_shape, 
                                     const double& I_A_rate,
                                     const double& I_B_shape,
                                     const double& I_B_rate,
                                     const double& sigma_A_mean,
                                     const double& sigma_A_shape,
                                     const double& sigma_B_mean,
                                     const double& sigma_B_shape,
                                     const arma::vec& eps_step){
  
  arma::vec grad = calc_gradient(Labels, transform_pars(theta), X_A, X_B, X_AB, n_A,
                                 n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                                 sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                                 eps_step);
  grad = grad + arma::ones(grad.n_elem);
  
  return(grad);
}

inline arma::vec trans_calc_gradient_TI(arma::field<arma::vec>& Labels,
                                        arma::vec theta,
                                        arma::vec& basis_coef_A,
                                        arma::vec& basis_coef_B,
                                        const arma::field<arma::mat>& basis_funct_A,
                                        const arma::field<arma::mat>& basis_funct_B,
                                        const arma::field<arma::mat>& basis_funct_AB,
                                        const arma::field<arma::vec>& X_A,
                                        const arma::field<arma::vec>& X_B,
                                        const arma::field<arma::vec>& X_AB,
                                        const arma::vec& n_A,
                                        const arma::vec& n_B,
                                        const arma::vec& n_AB,
                                        const double& mu_A, 
                                        const double& mu_B,
                                        const double& I_A_sigma_sq,
                                        const double& I_B_sigma_sq,
                                        const double& sigma_A_mean,
                                        const double& sigma_A_shape,
                                        const double& sigma_B_mean,
                                        const double& sigma_B_shape,
                                        const arma::mat P_mat,
                                        const double& eps_step){
  arma::vec grad = calc_gradient_TI(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                 basis_funct_A, basis_funct_B, basis_funct_AB,
                                 X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, 
                                 I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean, 
                                 sigma_A_shape,sigma_B_mean, sigma_B_shape,
                                 P_mat, eps_step);
  grad.subvec(basis_coef_A.n_elem + basis_coef_B.n_elem, grad.n_elem-1) = grad.subvec(basis_coef_A.n_elem + basis_coef_B.n_elem, grad.n_elem-1) + arma::ones(2);
  
  return(grad);
}

inline arma::vec trans_calc_gradient_FR(arma::field<arma::vec>& Labels,
                                        arma::vec theta,
                                        arma::vec& basis_coef_A,
                                        arma::vec& basis_coef_B,
                                        const arma::field<arma::mat>& basis_funct_A,
                                        const arma::field<arma::mat>& basis_funct_B,
                                        const arma::field<arma::mat>& basis_funct_AB,
                                        const arma::field<arma::vec>& X_A,
                                        const arma::field<arma::vec>& X_B,
                                        const arma::field<arma::vec>& X_AB,
                                        const arma::vec& n_A,
                                        const arma::vec& n_B,
                                        const arma::vec& n_AB,
                                        const double& mu_A, 
                                        const double& mu_B,
                                        const double& I_A_sigma_sq,
                                        const double& I_B_sigma_sq,
                                        const arma::mat P_mat,
                                        const double& eps_step){
  arma::vec grad = calc_gradient_FR(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                    basis_funct_A, basis_funct_B, basis_funct_AB,
                                    X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, 
                                    I_A_sigma_sq, I_B_sigma_sq,
                                    P_mat, eps_step);
  return(grad);
}

inline arma::vec trans_calc_gradient_sigma(arma::field<arma::vec>& Labels,
                                        arma::vec theta,
                                        arma::vec& basis_coef_A,
                                        arma::vec& basis_coef_B,
                                        const arma::field<arma::mat>& basis_funct_A,
                                        const arma::field<arma::mat>& basis_funct_B,
                                        const arma::field<arma::mat>& basis_funct_AB,
                                        const arma::field<arma::vec>& X_A,
                                        const arma::field<arma::vec>& X_B,
                                        const arma::field<arma::vec>& X_AB,
                                        const arma::vec& n_A,
                                        const arma::vec& n_B,
                                        const arma::vec& n_AB,
                                        const double& sigma_A_mean,
                                        const double& sigma_A_shape,
                                        const double& sigma_B_mean,
                                        const double& sigma_B_shape,
                                        const double& eps_step){
  arma::vec grad = calc_gradient_sigma(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                       basis_funct_A, basis_funct_B, basis_funct_AB,
                                       X_A, X_B, X_AB, n_A, n_B, n_AB, sigma_A_mean, 
                                       sigma_A_shape,sigma_B_mean, sigma_B_shape,
                                       eps_step);
  grad = grad + arma::ones(grad.n_elem);
  
  return(grad);
}

}


#endif