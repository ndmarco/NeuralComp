#ifndef NeuralComp_HMC_H
#define NeuralComp_HMC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Posterior.h"

namespace NeuralComp {

inline void leapfrog_step(arma::field<arma::vec>& Labels,
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
                          const arma::vec& eps_step,
                          arma::vec step_size,
                          arma::vec& position,
                          arma::vec& momentum){
  arma::vec momentum_i = momentum + 0.5 * arma::diagmat(step_size) * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        delta_shape, delta_rate, eps_step);
  position = position + arma::diagmat(step_size) * momentum;
  momentum = momentum_i + 0.5 * arma::diagmat(step_size) * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        delta_shape, delta_rate, eps_step);
}

inline void leapfrog(arma::field<arma::vec>& Labels,
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
                     const arma::vec& eps_step,
                     arma::vec& step_size,
                     arma::vec& position,
                     arma::vec& momentum,
                     arma::vec& prop_position,
                     arma::vec& prop_momentum,
                     int Leapfrog_steps){
  prop_position = position;
  prop_momentum = momentum;
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step(Labels, X_A, X_B, X_AB, n_A,
                  n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                  sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                  delta_shape, delta_rate, eps_step, step_size, prop_position,
                  prop_momentum);
  }
}

inline double lprob_accept(arma::vec& prop_position,
                           arma::vec& prop_momentum,
                           arma::vec& position,
                           arma::vec& momentum,
                           arma::field<arma::vec>& Labels,
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
  double lp_accept = transformed_log_posterior(Labels, prop_position, X_A, X_B, X_AB,
                                               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                               I_B_rate, sigma_A_mean, sigma_A_shape,
                                               sigma_B_mean, sigma_B_shape, delta_shape,
                                               delta_rate);
  lp_accept = lp_accept + arma::accu(arma::normpdf(prop_momentum));
  lp_accept = lp_accept - transformed_log_posterior(Labels, position, X_A, X_B, X_AB,
                                                    n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                                    I_B_rate, sigma_A_mean, sigma_A_shape,
                                                    sigma_B_mean, sigma_B_shape, delta_shape,
                                                    delta_rate);
  lp_accept = lp_accept - arma::accu(arma::normpdf(momentum));
  return lp_accept;
}


inline void HMC_step(arma::field<arma::vec>& Labels,
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
                     const arma::vec& eps_step,
                     arma::vec step_size,
                     int Leapfrog_steps,
                     double& num_accept){
  arma::vec momentum = arma::randn(theta.n_elem);
  arma::vec prop_position = theta;
  arma::vec prop_momentum = momentum;
  leapfrog(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, I_A_rate, 
           I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape, sigma_B_mean,
           sigma_B_shape, delta_shape, delta_rate, eps_step, step_size, 
           theta, momentum, prop_position, prop_momentum, Leapfrog_steps);
  double accept = lprob_accept(prop_position, prop_momentum, theta, momentum,
                               Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
                               I_A_rate, I_B_shape, I_B_rate, sigma_A_mean,
                               sigma_A_shape, sigma_B_mean, sigma_B_shape,
                               delta_shape, delta_rate);
  if(std::log(R::runif(0,1)) < accept){
    num_accept = num_accept + 1;
    theta = prop_position;
  }
}

inline arma::mat HMC_sampler(arma::field<arma::vec> Labels,
                             const arma::field<arma::vec> X_A,
                             const arma::field<arma::vec> X_B,
                             const arma::field<arma::vec> X_AB,
                             const arma::vec n_A,
                             const arma::vec n_B,
                             const arma::vec n_AB,
                             const arma::vec init_position,
                             int MCMC_iters,
                             int Leapfrog_steps,
                             const double I_A_shape, 
                             const double I_A_rate,
                             const double I_B_shape,
                             const double I_B_rate,
                             const double sigma_A_mean,
                             const double sigma_A_shape,
                             const double sigma_B_mean,
                             const double sigma_B_shape,
                             const double delta_shape,
                             const double delta_rate,
                             const arma::vec eps_step,
                             arma::vec step_size){
  arma::mat theta(MCMC_iters, init_position.n_elem, arma::fill::ones);
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  double num_accept = 0;
  for(int i = 1; i < MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "iteration = " << i << "\n";
      Rcpp::Rcout << "prob_accept = " << num_accept / i << "\n";
    }
    theta_ph = theta.row(i).t();
    HMC_step(Labels, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, step_size, Leapfrog_steps, num_accept);
    theta.row(i) = theta_ph.t();
    if((i+1) < MCMC_iters){
      theta.row(i+1) = theta.row(i);
    }
  }
  
  return arma::exp(theta);
}



}


#endif