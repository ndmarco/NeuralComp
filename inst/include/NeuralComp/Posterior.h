#ifndef NeuralComp_Posterior_H
#define NeuralComp_Posterior_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"

namespace NeuralComp {

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
                            const double& sigma_B_shape,
                            const double& delta_shape,
                            const double& delta_rate){
  double l_posterior = log_likelihood(Labels, theta, X_A, X_B, X_AB, n_A, n_B, n_AB) +
    log_prior(I_A_shape, I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
              sigma_B_mean, sigma_B_shape, delta_shape, delta_rate, theta);
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
                                        const double& sigma_B_shape,
                                        const double& delta_shape,
                                        const double& delta_rate){
  double l_posterior = log_posterior(Labels, transform_pars(theta), X_A, X_B, X_AB,
                                     n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                     I_B_rate, sigma_A_mean, sigma_A_shape,
                                     sigma_B_mean, sigma_B_shape, delta_shape,
                                     delta_rate) + arma::accu(theta);
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
                               const double& delta_shape,
                               const double& delta_rate,
                               const arma::vec& eps_step){
  arma::vec grad(theta.n_elem, arma::fill::zeros);
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  for(int i = 0; i < theta.n_elem; i++){
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
               sigma_B_mean, sigma_B_shape, delta_shape,
               delta_rate) - log_posterior(Labels, theta_m_eps, X_A, X_B, X_AB,
               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
               I_B_rate, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape, delta_shape,
               delta_rate)) / (2 * eps_step(i));
    
  }
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
                                     const double& delta_shape,
                                     const double& delta_rate,
                                     const arma::vec& eps_step){
  
  arma::vec grad = calc_gradient(Labels, transform_pars(theta), X_A, X_B, X_AB, n_A,
                                 n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                                 sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                                 delta_shape, delta_rate, eps_step);
  grad = grad + arma::ones(grad.n_elem);
  
  return(grad);
}

}


#endif