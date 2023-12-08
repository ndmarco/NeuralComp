#ifndef NeuralComp_HMC_H
#define NeuralComp_HMC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Posterior.h"
#include "Labels.h"
#include "SplinePrior.h"
#include <CppAD.h>
using namespace CppAD;
using namespace Eigen;

namespace NeuralComp {

inline void leapfrog_step(arma::field<arma::vec>& Labels,
                          const arma::field<arma::vec>& X_A,
                          const arma::field<arma::vec>& X_B,
                          const arma::field<arma::vec>& X_AB,
                          const arma::vec& n_A,
                          const arma::vec& n_B,
                          const arma::vec& n_AB,
                          const double& I_A_mean, 
                          const double& I_A_shape,
                          const double& I_B_mean,
                          const double& I_B_shape,
                          const double& sigma_A_mean,
                          const double& sigma_A_shape,
                          const double& sigma_B_mean,
                          const double& sigma_B_shape,
                          const arma::vec& eps_step,
                          double& step_size,
                          arma::mat& Mass_mat,
                          arma::vec& position,
                          arma::vec& momentum){
  arma::vec momentum_i = momentum + 0.5 * step_size * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        eps_step);
  position.subvec(0,3) = position.subvec(0,3) + step_size * arma::inv_sympd(Mass_mat) * momentum_i;
  momentum = momentum_i + 0.5 * step_size * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        eps_step);
}

inline void leapfrog_step_TI(arma::field<arma::vec>& Labels,
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
                             const double& eps_step,
                             double& step_size,
                             arma::vec& position,
                             arma::vec& momentum){
  arma::vec position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  arma::vec momentum_i = momentum + 0.5 * step_size * 
    trans_calc_gradient_TI(Labels, position_theta, position_basis_coef_A, 
                           position_basis_coef_B, basis_funct_A, basis_funct_B,
                           basis_funct_AB, X_A, X_B, X_AB, n_A,
                           n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq,
                           sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                           P_mat, eps_step);
  
  position.subvec(0,position.n_elem-2) = position.subvec(0,position.n_elem-2) + step_size * momentum_i;
  position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  momentum = momentum_i + 0.5 * step_size * 
    trans_calc_gradient_TI(Labels, position_theta, position_basis_coef_A, 
                           position_basis_coef_B, basis_funct_A, basis_funct_B,
                           basis_funct_AB, X_A, X_B, X_AB, n_A,
                           n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq,
                           sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                           P_mat, eps_step);
}

inline void leapfrog_step_FR(arma::field<arma::vec>& Labels,
                             const arma::field<arma::mat>& basis_funct_A,
                             const arma::field<arma::mat>& basis_funct_B,
                             const arma::field<arma::mat>& basis_funct_AB,
                             const arma::field<arma::vec>& X_A,
                             const arma::field<arma::vec>& X_B,
                             const arma::field<arma::vec>& X_AB,
                             const arma::vec& n_A,
                             const arma::vec& n_B,
                             const arma::vec& n_AB,
                             const double& I_A_sigma_sq,
                             const double& I_B_sigma_sq,
                             const double& eps_step,
                             double& step_size,
                             arma::mat& Mass_mat,
                             arma::vec& position,
                             arma::vec& momentum,
                             ADFun<double>& gr,
                             int step_num,
                             int num_leapfrog){
  position.subvec(0, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1) = position.subvec(0, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1) + 
    step_size * momentum;
  arma::vec position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  if(step_num != (num_leapfrog - 1)){
    momentum = momentum + step_size * 
      trans_calc_gradient_eigen_basis_update(Labels, position_theta, position_basis_coef_A, 
                                             position_basis_coef_B, basis_funct_A, basis_funct_B,
                                             basis_funct_AB, X_A, X_B, X_AB, n_A,
                                             n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, gr);
  }
}

inline void leapfrog_step_theta(arma::field<arma::vec>& Labels,
                                const arma::field<arma::mat>& basis_funct_A,
                                const arma::field<arma::mat>& basis_funct_B,
                                const arma::field<arma::mat>& basis_funct_AB,
                                const arma::field<arma::vec>& X_A,
                                const arma::field<arma::vec>& X_B,
                                const arma::field<arma::vec>& X_AB,
                                const arma::vec& n_A,
                                const arma::vec& n_B,
                                const arma::vec& n_AB,
                                const double& I_A_mean, 
                                const double& I_A_shape,
                                const double& I_B_mean,
                                const double& I_B_shape,
                                const double& sigma_A_mean,
                                const double& sigma_A_shape,
                                const double& sigma_B_mean,
                                const double& sigma_B_shape,
                                const double& eps_step,
                                double& step_size,
                                arma::mat& Mass_mat,
                                arma::vec& position,
                                arma::vec& momentum,
                                ADFun<double>& gr,
                                int step_num,
                                int num_leapfrog){
  // update position
  position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols,position.n_elem-2) = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols,position.n_elem-2) + 
    step_size * momentum;
  arma::vec position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  // update momentum
  if(step_num != (num_leapfrog - 1)){
    momentum = momentum + step_size * 
      trans_calc_gradient_eigen_theta_update(Labels, position_theta, position_basis_coef_A, 
                                             position_basis_coef_B, basis_funct_A, basis_funct_B,
                                             basis_funct_AB, X_A, X_B, X_AB, n_A,
                                             n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                             sigma_A_mean, sigma_A_shape, 
                                             sigma_B_mean, sigma_B_shape, gr);
  }
  
}




inline void leapfrog_step_delta(arma::field<arma::vec>& Labels,
                                const arma::field<arma::vec>& X_A,
                                const arma::field<arma::vec>& X_B,
                                const arma::field<arma::vec>& X_AB,
                                const arma::vec& n_A,
                                const arma::vec& n_B,
                                const arma::vec& n_AB,
                                const double& delta_shape,
                                const double& delta_rate,
                                const arma::vec& eps_step,
                                double& step_size,
                                arma::vec& position,
                                double& momentum){
  double momentum_i = momentum + 0.5 * step_size * 
    trans_calc_gradient_delta(Labels, position, X_A, X_B, X_AB, n_A,
                              n_B, n_AB, delta_shape, delta_rate,
                              eps_step);
  position(4) = position(4) + step_size * momentum;
  momentum = momentum_i + 0.5 * step_size * 
    trans_calc_gradient_delta(Labels, position, X_A, X_B, X_AB, n_A,
                              n_B, n_AB, delta_shape, delta_rate,
                              eps_step);
}

inline void leapfrog(arma::field<arma::vec>& Labels,
                     const arma::field<arma::vec>& X_A,
                     const arma::field<arma::vec>& X_B,
                     const arma::field<arma::vec>& X_AB,
                     const arma::vec& n_A,
                     const arma::vec& n_B,
                     const arma::vec& n_AB,
                     const double& I_A_mean, 
                     const double& I_A_shape,
                     const double& I_B_mean,
                     const double& I_B_shape,
                     const double& sigma_A_mean,
                     const double& sigma_A_shape,
                     const double& sigma_B_mean,
                     const double& sigma_B_shape,
                     const arma::vec& eps_step,
                     double& step_size,
                     arma::mat& Mass_mat,
                     arma::vec& position,
                     arma::vec& momentum,
                     arma::vec& prop_position,
                     arma::vec& prop_momentum,
                     int Leapfrog_steps){
  prop_position = position;
  prop_momentum = momentum;
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step(Labels, X_A, X_B, X_AB, n_A,
                  n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                  sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                  eps_step, step_size, Mass_mat,
                  prop_position, prop_momentum);
  }
}

inline void leapfrog_TI(arma::field<arma::vec>& Labels,
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
                        const double& eps_step,
                        double& step_size,
                        arma::vec& position,
                        arma::vec& momentum,
                        arma::vec& prop_position,
                        arma::vec& prop_momentum,
                        int Leapfrog_steps){
  prop_position = position;
  prop_momentum = momentum;
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_TI(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, 
                     X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq,
                     sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                     P_mat, eps_step, step_size, prop_position, prop_momentum);
  }
}

inline void leapfrog_FR(arma::field<arma::vec>& Labels,
                        const arma::field<arma::mat>& basis_funct_A,
                        const arma::field<arma::mat>& basis_funct_B,
                        const arma::field<arma::mat>& basis_funct_AB,
                        const arma::field<arma::vec>& X_A,
                        const arma::field<arma::vec>& X_B,
                        const arma::field<arma::vec>& X_AB,
                        const arma::vec& n_A,
                        const arma::vec& n_B,
                        const arma::vec& n_AB,
                        const double& I_A_sigma_sq,
                        const double& I_B_sigma_sq,
                        const double& eps_step,
                        double& step_size,
                        arma::mat& Mass_mat,
                        arma::vec& position,
                        arma::vec& momentum,
                        arma::vec& prop_position,
                        arma::vec& prop_momentum,
                        int Leapfrog_steps){
  ADFun<double> gr;
  prop_position = position;
  prop_momentum = momentum;
  arma::vec position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  // Initial half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_basis(Labels, position_theta, position_basis_coef_A, 
                                    position_basis_coef_B, basis_funct_A, basis_funct_B,
                                    basis_funct_AB, X_A, X_B, X_AB, n_A,
                                    n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, gr);
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_FR(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, 
                     X_B, X_AB, n_A, n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq,
                     eps_step, step_size, Mass_mat, prop_position, prop_momentum, gr, i, Leapfrog_steps);
  }
  position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  // Final half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_basis_update(Labels, position_theta, position_basis_coef_A, 
                                           position_basis_coef_B, basis_funct_A, basis_funct_B,
                                           basis_funct_AB, X_A, X_B, X_AB, n_A,
                                           n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, gr);
}

inline void leapfrog_theta(arma::field<arma::vec>& Labels,
                           const arma::field<arma::mat>& basis_funct_A,
                           const arma::field<arma::mat>& basis_funct_B,
                           const arma::field<arma::mat>& basis_funct_AB,
                           const arma::field<arma::vec>& X_A,
                           const arma::field<arma::vec>& X_B,
                           const arma::field<arma::vec>& X_AB,
                           const arma::vec& n_A,
                           const arma::vec& n_B,
                           const arma::vec& n_AB,
                           const double& I_A_mean, 
                           const double& I_A_shape,
                           const double& I_B_mean,
                           const double& I_B_shape,
                           const double& sigma_A_mean,
                           const double& sigma_A_shape,
                           const double& sigma_B_mean,
                           const double& sigma_B_shape,
                           const double& eps_step,
                           double& step_size,
                           arma::mat& Mass_mat,
                           arma::vec& position,
                           arma::vec& momentum,
                           arma::vec& prop_position,
                           arma::vec& prop_momentum,
                           int Leapfrog_steps){
  ADFun<double> gr;
  prop_position = position;
  prop_momentum = momentum;
  arma::vec position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  // Initial half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_theta(Labels, position_theta, position_basis_coef_A, 
                                    position_basis_coef_B, basis_funct_A, basis_funct_B,
                                    basis_funct_AB, X_A, X_B, X_AB, n_A,
                                    n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                    sigma_A_mean, sigma_A_shape, 
                                    sigma_B_mean, sigma_B_shape, gr);
  
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_theta(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, 
                        X_B, X_AB, n_A, n_B, n_AB,
                        sigma_A_mean, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                        sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        eps_step, step_size, Mass_mat, prop_position, prop_momentum, gr, i, Leapfrog_steps);
  }
  
  position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  // Final half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_theta_update(Labels, position_theta, position_basis_coef_A, 
                                           position_basis_coef_B, basis_funct_A, basis_funct_B,
                                           basis_funct_AB, X_A, X_B, X_AB, n_A,
                                           n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                           sigma_A_mean, sigma_A_shape, 
                                           sigma_B_mean, sigma_B_shape, gr);
}

inline void leapfrog_delta(arma::field<arma::vec>& Labels,
                           const arma::field<arma::vec>& X_A,
                           const arma::field<arma::vec>& X_B,
                           const arma::field<arma::vec>& X_AB,
                           const arma::vec& n_A,
                           const arma::vec& n_B,
                           const arma::vec& n_AB,
                           const double& delta_shape,
                           const double& delta_rate,
                           const arma::vec& eps_step,
                           double& step_size,
                           arma::vec& position,
                           double& momentum,
                           arma::vec& prop_position,
                           double& prop_momentum,
                           int Leapfrog_steps){
  prop_position = position;
  prop_momentum = momentum;
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_delta(Labels, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, delta_shape, delta_rate,
                        eps_step, step_size,
                        prop_position, prop_momentum);
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
                           arma::mat& Mass_mat){
  double lp_accept = transformed_log_posterior(Labels, prop_position, X_A, X_B, X_AB,
                                               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                               I_B_rate, sigma_A_mean, sigma_A_shape,
                                               sigma_B_mean, sigma_B_shape);
  lp_accept = lp_accept + 0.5 * arma::dot(arma::solve(Mass_mat, prop_momentum), prop_momentum);
  lp_accept = lp_accept - transformed_log_posterior(Labels, position, X_A, X_B, X_AB,
                                                    n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
                                                    I_B_rate, sigma_A_mean, sigma_A_shape,
                                                    sigma_B_mean, sigma_B_shape);
  lp_accept = lp_accept - 0.5 * arma::dot(arma::solve(Mass_mat, momentum), momentum);
  return lp_accept;
}

inline double lprob_accept_TI(arma::vec& prop_position,
                              arma::vec& prop_momentum,
                              arma::vec& position,
                              arma::vec& momentum,
                              arma::field<arma::vec>& Labels,
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
  arma::vec position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  
  double lp_accept = transformed_log_posterior_TI(Labels, position_theta, position_basis_coef_A,
                                                  position_basis_coef_B, basis_funct_A,
                                                  basis_funct_B, basis_funct_AB,
                                                  X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A,
                                                  mu_B, I_A_sigma_sq, I_B_sigma_sq, 
                                                  sigma_A_mean, sigma_A_shape,
                                                  sigma_B_mean, sigma_B_shape, P_mat);
  
  lp_accept = lp_accept + 0.5 * arma::dot(prop_momentum, prop_momentum);
  
  position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);

  lp_accept = lp_accept - transformed_log_posterior_TI(Labels, position_theta, position_basis_coef_A,
                                                       position_basis_coef_B, basis_funct_A,
                                                       basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
                                                       n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, 
                                                       I_B_sigma_sq, sigma_A_mean, sigma_A_shape,
                                                       sigma_B_mean, sigma_B_shape, P_mat);
  lp_accept = lp_accept - 0.5 * arma::dot(momentum, momentum);
  return lp_accept;
}

inline double lprob_accept_FR(arma::vec& prop_position,
                              arma::vec& prop_momentum,
                              arma::vec& position,
                              arma::vec& momentum,
                              arma::field<arma::vec>& Labels,
                              const arma::field<arma::mat>& basis_funct_A,
                              const arma::field<arma::mat>& basis_funct_B,
                              const arma::field<arma::mat>& basis_funct_AB,
                              const arma::field<arma::vec>& X_A,
                              const arma::field<arma::vec>& X_B,
                              const arma::field<arma::vec>& X_AB,
                              const arma::vec& n_A,
                              const arma::vec& n_B,
                              const arma::vec& n_AB,
                              const double& I_A_sigma_sq,
                              const double& I_B_sigma_sq,
                              arma::mat& Mass_mat){
  arma::vec position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  
  double lp_accept = transformed_log_posterior_FR(Labels, position_theta, position_basis_coef_A,
                                                  position_basis_coef_B, basis_funct_A,
                                                  basis_funct_B, basis_funct_AB,
                                                  X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                  I_A_sigma_sq, I_B_sigma_sq);
  
  lp_accept = lp_accept - 0.5 * arma::dot(arma::solve(Mass_mat, prop_momentum), prop_momentum);
  
  position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  
  lp_accept = lp_accept - transformed_log_posterior_FR(Labels, position_theta, position_basis_coef_A,
                                                       position_basis_coef_B, basis_funct_A,
                                                       basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
                                                       n_A, n_B, n_AB, I_A_sigma_sq, 
                                                       I_B_sigma_sq);
  lp_accept = lp_accept + 0.5 * arma::dot(arma::solve(Mass_mat, momentum), momentum);
  return lp_accept;
}

inline double lprob_accept_theta(arma::vec& prop_position,
                                 arma::vec& prop_momentum,
                                 arma::vec& position,
                                 arma::vec& momentum,
                                 arma::field<arma::vec>& Labels,
                                 const arma::field<arma::mat>& basis_funct_A,
                                 const arma::field<arma::mat>& basis_funct_B,
                                 const arma::field<arma::mat>& basis_funct_AB,
                                 const arma::field<arma::vec>& X_A,
                                 const arma::field<arma::vec>& X_B,
                                 const arma::field<arma::vec>& X_AB,
                                 const arma::vec& n_A,
                                 const arma::vec& n_B,
                                 const arma::vec& n_AB,
                                 const double& I_A_mean, 
                                 const double& I_A_shape,
                                 const double& I_B_mean,
                                 const double& I_B_shape,
                                 const double& sigma_A_mean,
                                 const double& sigma_A_shape,
                                 const double& sigma_B_mean,
                                 const double& sigma_B_shape,
                                 arma::mat& Mass_mat){
  arma::vec position_theta = prop_position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = prop_position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = prop_position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  
  double lp_accept = transformed_log_posterior_theta(Labels, position_theta, position_basis_coef_A,
                                                     position_basis_coef_B, basis_funct_A,
                                                     basis_funct_B, basis_funct_AB,
                                                     X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                     I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                                     sigma_A_mean, sigma_A_shape,
                                                     sigma_B_mean, sigma_B_shape);
  
  lp_accept = lp_accept - 0.5 * arma::dot(arma::solve(Mass_mat, prop_momentum), prop_momentum);
  
  position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  
  lp_accept = lp_accept - transformed_log_posterior_theta(Labels, position_theta, position_basis_coef_A,
                                                          position_basis_coef_B, basis_funct_A,
                                                          basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
                                                          n_A, n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                                          sigma_A_mean, sigma_A_shape,
                                                          sigma_B_mean, sigma_B_shape);
  lp_accept = lp_accept + 0.5 * arma::dot(arma::solve(Mass_mat, momentum), momentum);
  return lp_accept;
}

inline double lprob_accept_delta(arma::vec& prop_position,
                                 double& prop_momentum,
                                 arma::vec& position,
                                 double& momentum,
                                 arma::field<arma::vec>& Labels,
                                 const arma::field<arma::vec>& X_A,
                                 const arma::field<arma::vec>& X_B,
                                 const arma::field<arma::vec>& X_AB,
                                 const arma::vec& n_A,
                                 const arma::vec& n_B,
                                 const arma::vec& n_AB,
                                 const double& delta_shape,
                                 const double& delta_rate){
  double lp_accept = transformed_log_posterior_delta(Labels, prop_position, X_A, X_B, X_AB,
                                                     n_A, n_B, n_AB, delta_shape, delta_rate);
  lp_accept = lp_accept + 0.5 * pow(prop_momentum, 2.0);
  lp_accept = lp_accept - transformed_log_posterior_delta(Labels, position, X_A, X_B, X_AB,
                                                          n_A, n_B, n_AB, delta_shape, delta_rate);
  lp_accept = lp_accept - 0.5 * pow(momentum, 2.0);
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
                     arma::mat& Mass_mat,
                     double& step_size,
                     double& step_size_delta,
                     int Leapfrog_steps,
                     double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = arma::mvnrnd(arma::zeros(theta.n_elem - 1), Mass_mat);
  arma::vec prop_position = theta;
  arma::vec prop_momentum = momentum;
  leapfrog(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, I_A_rate,
           I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape, sigma_B_mean,
           sigma_B_shape, eps_step, step_size, Mass_mat,
           theta, momentum, prop_position, prop_momentum, Leapfrog_steps);
  double accept = lprob_accept(prop_position, prop_momentum, theta, momentum,
                               Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape,
                               I_A_rate, I_B_shape, I_B_rate, sigma_A_mean,
                               sigma_A_shape, sigma_B_mean, sigma_B_shape,
                               Mass_mat);
  
  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    theta = prop_position;
  }
  
  // // Sample for delta
  // double momentum_delta = R::rnorm(0,1);
  // prop_position = theta;
  // double prop_momentum_delta = momentum_delta;
  // leapfrog_delta(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, delta_shape, delta_rate,
  //                eps_step, step_size_delta, theta, momentum_delta, prop_position,
  //                prop_momentum_delta, Leapfrog_steps);
  // double accept_delta = lprob_accept_delta(prop_position, prop_momentum_delta, theta,
  //                                          momentum_delta, Labels, X_A, X_B, X_AB, n_A,
  //                                          n_B, n_AB, delta_shape, delta_rate);
  // if(std::log(R::runif(0,1)) < accept_delta){
  //   num_accept_delta = 1;
  //   theta = prop_position;
  // }
  
}

inline void HMC_step_TI(arma::field<arma::vec>& Labels,
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
                        const double& eps_step,
                        double& step_size,
                        int Leapfrog_steps,
                        double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = Rcpp::rnorm(basis_coef_A.n_elem + basis_coef_B.n_elem + theta.n_elem - 1, 0, 1);
  arma::vec prop_position = arma::zeros(basis_coef_A.n_elem + basis_coef_B.n_elem + theta.n_elem);
  prop_position.subvec(0, basis_coef_A.n_elem -1) = basis_coef_A;
  prop_position.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1) = basis_coef_B;
  prop_position.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, prop_position.n_elem - 1) = theta;
  arma::vec position = prop_position;
  arma::vec prop_momentum = momentum;
  leapfrog_TI(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
              n_A, n_B, n_AB, mu_A, mu_B, I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean,
              sigma_A_shape, sigma_B_mean, sigma_B_shape, P_mat, eps_step, step_size, 
              position, momentum, prop_position, prop_momentum, Leapfrog_steps);

  double accept = lprob_accept_TI(prop_position, prop_momentum, position, momentum,
                                  Labels, basis_funct_A, basis_funct_B, basis_funct_AB,
                                  X_A, X_B, X_AB, n_A, n_B, n_AB, mu_A, mu_B, 
                                  I_A_sigma_sq, I_B_sigma_sq, sigma_A_mean,
                                  sigma_A_shape, sigma_B_mean, sigma_B_shape, P_mat);

  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    theta = prop_position.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, prop_position.n_elem - 1);
    basis_coef_A = prop_position.subvec(0, basis_coef_A.n_elem -1);
    basis_coef_B = prop_position.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
  }
  
}

inline void HMC_step_FR(arma::field<arma::vec>& Labels,
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
                        const arma::vec& n_AB,
                        const double& I_A_sigma_sq,
                        const double& I_B_sigma_sq,
                        const double& eps_step,
                        double& step_size,
                        arma::mat& Mass_mat,
                        int Leapfrog_steps,
                        double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = arma::mvnrnd(arma::zeros(basis_coef_A.n_elem + basis_coef_B.n_elem), Mass_mat);
  arma::vec prop_position = arma::zeros(basis_coef_A.n_elem + basis_coef_B.n_elem + theta.n_elem);
  prop_position.subvec(0, basis_coef_A.n_elem -1) = basis_coef_A;
  prop_position.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1) = basis_coef_B;
  prop_position.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, prop_position.n_elem - 1) = theta;
  arma::vec position = prop_position;
  arma::vec prop_momentum = momentum;
  leapfrog_FR(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
              n_A, n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, eps_step, step_size, Mass_mat,
              position, momentum, prop_position, prop_momentum, Leapfrog_steps);
  double accept = lprob_accept_FR(prop_position, prop_momentum, position, momentum,
                                  Labels, basis_funct_A, basis_funct_B, basis_funct_AB,
                                  X_A, X_B, X_AB, n_A, n_B, n_AB,
                                  I_A_sigma_sq, I_B_sigma_sq, Mass_mat);
  
  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    basis_coef_A = prop_position.subvec(0, basis_coef_A.n_elem -1);
    basis_coef_B = prop_position.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1);
  }
  
}



inline void HMC_step_theta(arma::field<arma::vec>& Labels,
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
                           const arma::vec& n_AB,
                           const double& I_A_mean, 
                           const double& I_A_shape,
                           const double& I_B_mean,
                           const double& I_B_shape,
                           const double& sigma_A_mean,
                           const double& sigma_A_shape,
                           const double& sigma_B_mean,
                           const double& sigma_B_shape,
                           const double& eps_step,
                           double& step_size,
                           arma::mat& Mass_mat,
                           int Leapfrog_steps,
                           double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = arma::mvnrnd(arma::zeros(theta.n_elem - 1), Mass_mat);
  arma::vec prop_position = arma::zeros(basis_coef_A.n_elem + basis_coef_B.n_elem + theta.n_elem);
  prop_position.subvec(0, basis_coef_A.n_elem -1) = basis_coef_A;
  prop_position.subvec(basis_coef_A.n_elem, basis_coef_B.n_elem + basis_coef_A.n_elem - 1) = basis_coef_B;
  prop_position.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, prop_position.n_elem - 1) = theta;
  arma::vec position = prop_position;
  arma::vec prop_momentum = momentum;
  leapfrog_theta(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, X_B, X_AB,
                 n_A, n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                 sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                 eps_step, step_size, Mass_mat, position, momentum, prop_position,
                 prop_momentum, Leapfrog_steps);
  
  double accept = lprob_accept_theta(prop_position, prop_momentum, position, momentum,
                                     Labels, basis_funct_A, basis_funct_B, basis_funct_AB,
                                     X_A, X_B, X_AB, n_A, n_B, n_AB,
                                     I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                     sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape, Mass_mat);
  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    theta = prop_position.subvec(basis_coef_B.n_elem + basis_coef_A.n_elem, prop_position.n_elem - 1);
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
                             double& step_size,
                             double& step_size_delta,
                             arma::mat& Mass_mat,
                             int Warm_block){
  arma::mat theta(MCMC_iters + Warm_block, init_position.n_elem, arma::fill::ones);
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec vec_accept(MCMC_iters + Warm_block, arma::fill::zeros);
  arma::vec vec_accept_delta(MCMC_iters + Warm_block, arma::fill::zeros);
  double prop_accept_10 = 0;
  double prop_accept_10_delta = 0;
  for(int i = 1; i < Warm_block; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size =" << step_size << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    HMC_step(Labels, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps,
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
    }
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }

      // // adjust step size for delta
      // prop_accept_10_delta = arma::accu(vec_accept_delta.subvec(i-9, i))/ 10;
      // if(prop_accept_10_delta  <= 0.1){
      //   step_size_delta = step_size_delta * 0.1;
      // }else if(prop_accept_10_delta <= 0.3){
      //   step_size_delta = step_size_delta * 0.5;
      // }else if(prop_accept_10_delta <= 0.6){
      //   step_size_delta = step_size_delta * 0.8;
      // }else if(prop_accept_10_delta > 0.85){
      //   step_size_delta = step_size_delta * 1.5;
      // }
    }
  }
  
  for(int i =  Warm_block; i <  Warm_block + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    HMC_step(Labels, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
    }
  }
  
  return arma::exp(theta);
}

inline Rcpp::List HMC_sampler_TI(arma::field<arma::vec> Labels,
                                 const arma::field<arma::mat>& basis_funct_A,
                                 const arma::field<arma::mat>& basis_funct_B,
                                 const arma::field<arma::mat>& basis_funct_AB,
                                 const arma::field<arma::vec> X_A,
                                 const arma::field<arma::vec> X_B,
                                 const arma::field<arma::vec> X_AB,
                                 const arma::vec n_A,
                                 const arma::vec n_B,
                                 const arma::vec n_AB,
                                 int MCMC_iters,
                                 int Leapfrog_steps,
                                 const double sigma_A_mean,
                                 const double sigma_A_shape,
                                 const double sigma_B_mean,
                                 const double sigma_B_shape,
                                 const double alpha,
                                 const double beta,
                                 const double mu_prior_mean,
                                 const double mu_prior_var,
                                 const double eps_step,
                                 double& step_size,
                                 int Warm_block){
  arma::mat theta(MCMC_iters + Warm_block, 3, arma::fill::ones);
  arma::mat basis_coef_A(MCMC_iters + Warm_block, basis_funct_A(0,0).n_cols, arma::fill::ones);
  arma::mat basis_coef_B(MCMC_iters + Warm_block, basis_funct_B(0,0).n_cols, arma::fill::ones);
  arma::vec mu_A(MCMC_iters + Warm_block, arma::fill::ones);
  basis_coef_A = basis_coef_A * mu_prior_mean;
  mu_A = mu_A * mu_prior_mean;
  arma::vec mu_B(MCMC_iters + Warm_block, arma::fill::ones);
  basis_coef_B = basis_coef_B * mu_prior_mean;
  mu_B = mu_B * mu_prior_mean;
  arma::vec I_A_sigma_sq(MCMC_iters + Warm_block, arma::fill::ones);
  arma::vec I_B_sigma_sq(MCMC_iters + Warm_block, arma::fill::ones);
  I_A_sigma_sq = I_A_sigma_sq;
  I_B_sigma_sq = I_B_sigma_sq;
  arma::vec init_position(3, arma::fill::ones);
  init_position(0) = std::sqrt(mu_prior_mean);
  init_position(1) = std::sqrt(mu_prior_mean);
  init_position(2) = 0.08;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec basis_coef_A_ph = basis_coef_A.row(0).t();
  arma::vec basis_coef_B_ph = basis_coef_B.row(0).t();
  arma::vec vec_accept(MCMC_iters + Warm_block, arma::fill::zeros);
  double prop_accept_10 = 0;
  double prop_accept_10_delta = 0;
  arma::mat P_mat(basis_coef_A.n_cols, basis_coef_A.n_cols, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }
  Rcpp::Rcout << P_mat;
  arma::vec theta_exp;
  arma::vec Basis_A_exp;
  arma::vec Basis_B_exp;
  for(int i = 1; i < Warm_block; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Step Size =" << step_size << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();

    HMC_step_TI(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                mu_A(i), mu_B(i), I_A_sigma_sq(i), I_B_sigma_sq(i), sigma_A_mean, sigma_A_shape,
                sigma_B_mean, sigma_B_shape,
                P_mat, eps_step, step_size, Leapfrog_steps, vec_accept(i));
    theta_exp = transform_pars(theta_ph);
    
    
    // update mu hyperparameters
    update_mu(basis_coef_A_ph, I_A_sigma_sq(i), mu_prior_mean, mu_prior_var,
              i, mu_A);
    update_mu(basis_coef_B_ph, I_B_sigma_sq(i), mu_prior_mean, mu_prior_var,
              i, mu_B);
    // update sigma hyperparameters
    update_I_sigma(basis_coef_A_ph, mu_A(i), alpha, beta, i, I_A_sigma_sq);
    update_I_sigma(basis_coef_B_ph, mu_B(i), alpha, beta, i, I_B_sigma_sq);
    
    Rcpp::Rcout << " log_lik = " << log_likelihood_TI(Labels, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
                                     basis_funct_A, basis_funct_B, basis_funct_AB,
                                     X_A, X_B, X_AB, n_A, n_B, n_AB);
    Rcpp::Rcout << " lprior = " << log_prior_TI(mu_A(i), mu_B(i), I_A_sigma_sq(i), I_B_sigma_sq(i), sigma_A_mean, sigma_A_shape,
                                      sigma_B_mean, sigma_B_shape, P_mat, theta_exp, basis_coef_A_ph, basis_coef_B_ph);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      mu_A(i + 1) = mu_A(i);
      mu_B(i + 1) = mu_B(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
    }
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block; i <  Warm_block + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    HMC_step_TI(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                mu_A(i), mu_B(i), I_A_sigma_sq(i), I_B_sigma_sq(i), sigma_A_mean, sigma_A_shape,
                sigma_B_mean, sigma_B_shape,
                P_mat, eps_step, step_size, Leapfrog_steps, vec_accept(i));
    theta_exp = transform_pars(theta_ph);
    
    
    // update mu hyperparameters
    update_mu(basis_coef_A_ph, I_A_sigma_sq(i), mu_prior_mean, mu_prior_var,
              i, mu_A);
    update_mu(basis_coef_B_ph,  I_B_sigma_sq(i), mu_prior_mean, mu_prior_var,
              i, mu_B);
    // update sigma hyperparameters
    update_I_sigma(basis_coef_A_ph, mu_A(i), alpha, beta, i, I_A_sigma_sq);
    update_I_sigma(basis_coef_B_ph, mu_B(i), alpha, beta, i, I_B_sigma_sq);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      mu_A(i + 1) = mu_A(i);
      mu_B(i + 1) = mu_B(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("basis_coef_A", basis_coef_A),
                                         Rcpp::Named("basis_coef_B", basis_coef_B),
                                         Rcpp::Named("mu_A", mu_A),
                                         Rcpp::Named("mu_B", mu_B),
                                         Rcpp::Named("I_A_sigma_sq", I_A_sigma_sq),
                                         Rcpp::Named("I_B_sigma_sq", I_B_sigma_sq));
  return params;
}


inline Rcpp::List HMC_sampler_FR(arma::field<arma::vec> Labels,
                                 const arma::field<arma::mat>& basis_funct_A,
                                 const arma::field<arma::mat>& basis_funct_B,
                                 const arma::field<arma::mat>& basis_funct_AB,
                                 const arma::field<arma::vec> X_A,
                                 const arma::field<arma::vec> X_B,
                                 const arma::field<arma::vec> X_AB,
                                 const arma::vec n_A,
                                 const arma::vec n_B,
                                 const arma::vec n_AB,
                                 int MCMC_iters,
                                 int Leapfrog_steps,
                                 const double& I_A_mean, 
                                 const double& I_A_shape,
                                 const double& I_B_mean,
                                 const double& I_B_shape,
                                 const double sigma_A_mean,
                                 const double sigma_A_shape,
                                 const double sigma_B_mean,
                                 const double sigma_B_shape,
                                 const double alpha,
                                 const double beta,
                                 const double eps_step,
                                 double& step_size_FR,
                                 double& step_size_theta,
                                 int Warm_block1,
                                 int Warm_block2){
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, 5, arma::fill::ones);
  arma::mat basis_coef_A(MCMC_iters + Warm_block1 + Warm_block2, basis_funct_A(0,0).n_cols, arma::fill::zeros);
  arma::mat basis_coef_B(MCMC_iters + Warm_block1 + Warm_block2, basis_funct_B(0,0).n_cols, arma::fill::zeros);
  arma::vec I_A_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec I_B_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec llik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  I_A_sigma_sq = I_A_sigma_sq;
  I_B_sigma_sq = I_B_sigma_sq;
  arma::vec init_position(5, arma::fill::ones);
  init_position(0) = I_A_mean;
  init_position(1) = I_B_mean;
  init_position(2) = sigma_A_mean;
  init_position(3) = sigma_B_mean;
  init_position(4) = 0.08;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec basis_coef_A_ph = basis_coef_A.row(0).t();
  arma::vec basis_coef_B_ph = basis_coef_B.row(0).t();
  arma::vec vec_accept_FR(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec vec_accept_sigma(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::mat Mass_mat_theta = arma::diagmat(arma::ones(theta.n_cols-1));
  arma::mat Mass_mat_basis = arma::diagmat(arma::ones(basis_coef_A.n_cols + basis_coef_B.n_cols));
  double prop_accept_10 = 0;
  double prop_accept_10_sigma = 0;
  
  arma::vec theta_exp;
  arma::vec Basis_A_exp;
  arma::vec Basis_B_exp;
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR= " << arma::accu(vec_accept_FR.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Prob_accept sigma= " << arma::accu(vec_accept_sigma.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Step Size theta =" << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR =" << step_size_FR << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    HMC_step_FR(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB,
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                eps_step, step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    
    HMC_step_theta(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   eps_step, step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_sigma(i));
    theta_exp = transform_pars(theta_ph);
    
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_A_ph, 0, alpha, beta, i, I_A_sigma_sq);
    update_I_sigma(basis_coef_B_ph, 0, alpha, beta, i, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
    }
    if((i % 10) == 0){
        // adjust step size for I_A, I_B, sigma_A, sigma_B
        prop_accept_10 = arma::accu(vec_accept_FR.subvec(i-9, i))/ 10;
        if(prop_accept_10  <= 0.1){
          step_size_FR = step_size_FR * 0.1;
        }else if(prop_accept_10 <= 0.3){
          step_size_FR = step_size_FR * 0.5;
        }else if(prop_accept_10 <= 0.6){
          step_size_FR = step_size_FR * 0.8;
        }else if(prop_accept_10 > 0.85){
          step_size_FR = step_size_FR * 2;
        }else if(prop_accept_10 > 0.9){
          step_size_FR = step_size_FR * 5;
        }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_sigma = arma::accu(vec_accept_sigma.subvec(i-9, i))/ 10;
      if(prop_accept_10_sigma  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_sigma <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_sigma <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_sigma > 0.85){
        step_size_theta = step_size_theta * 2;
      }else if(prop_accept_10_sigma > 0.9){
        step_size_theta = step_size_theta * 5;
      }
    }
  }
  Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(Warm_block1 - std::floor(0.5 *Warm_block1),0, Warm_block1 -1, theta.n_cols - 2)));
  arma::mat ph_basis = arma::zeros(std::ceil(0.5 *Warm_block1), basis_coef_A.n_cols + basis_coef_B.n_cols);
  ph_basis.submat(0, 0, std::ceil(0.5 *Warm_block1) - 1, basis_coef_A.n_cols-1) = basis_coef_A.submat(Warm_block1 - std::floor(0.5 *Warm_block1), 0, 
                  Warm_block1 -1, basis_coef_A.n_cols - 1);
  ph_basis.submat(0, basis_coef_A.n_cols, std::ceil(0.5 *Warm_block1) - 1, basis_coef_A.n_cols + basis_coef_B.n_cols - 1) = basis_coef_B.submat(Warm_block1 - std::floor(0.5 *Warm_block1), 0, 
                  Warm_block1 -1, basis_coef_B.n_cols - 1);
  Mass_mat_basis = arma::inv_sympd(arma::cov(ph_basis));
  // step_size_theta = step_size_theta / (arma::trace(Mass_mat_theta) / theta.n_cols -1);
  // step_size_FR = step_size_FR / (arma::trace(Mass_mat_basis) / basis_coef_A.n_cols + basis_coef_B.n_cols);
  // Rcpp::Rcout << " Mass_mat_basis " << Mass_mat_theta;
  for(int i = Warm_block1; i < Warm_block1 + Warm_block2; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR= " << arma::accu(vec_accept_FR.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept sigma= " << arma::accu(vec_accept_sigma.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta =" << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR =" << step_size_FR << "\n" << "\n";
    }
    if(i > Warm_block1){
      if((i % 500) == 0){
        Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(i - 500,0, i-1, theta.n_cols - 2)));
        arma::mat ph_basis1 = arma::zeros(i-500, basis_coef_A.n_cols + basis_coef_B.n_cols);
        ph_basis1.submat(0, 0, 499, basis_coef_A.n_cols-1) = basis_coef_A.submat(i-500, 0, 
                        i -1, basis_coef_A.n_cols - 1);
        ph_basis1.submat(0, basis_coef_A.n_cols, 499, basis_coef_A.n_cols + basis_coef_B.n_cols - 1) = basis_coef_B.submat(i-500, 0, 
                        i -1, basis_coef_B.n_cols - 1);
        Mass_mat_basis = arma::inv_sympd(arma::cov(ph_basis1));
      }
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    HMC_step_FR(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                eps_step, step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    HMC_step_theta(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   eps_step, step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_sigma(i));
    theta_exp = transform_pars(theta_ph);
    
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_A_ph, 0, alpha, beta, i, I_A_sigma_sq);
    update_I_sigma(basis_coef_B_ph, 0, alpha, beta, i, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
    }
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept_FR.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size_FR = step_size_FR * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size_FR = step_size_FR * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size_FR = step_size_FR * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size_FR = step_size_FR * 1.5;
      }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_sigma = arma::accu(vec_accept_sigma.subvec(i-9, i))/ 10;
      if(prop_accept_10_sigma  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_sigma <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_sigma <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_sigma > 0.85){
        step_size_theta = step_size_theta * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept FR= " << arma::accu(vec_accept_FR.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept sigma= " << arma::accu(vec_accept_sigma.subvec(i-50, i-1)) / 50 << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    HMC_step_FR(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                eps_step, step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    HMC_step_theta(Labels, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   eps_step, step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_sigma(i));
    theta_exp = transform_pars(theta_ph);
    
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_A_ph, 0, alpha, beta, i, I_A_sigma_sq);
    update_I_sigma(basis_coef_B_ph, 0, alpha, beta, i, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("basis_coef_A", basis_coef_A),
                                         Rcpp::Named("basis_coef_B", basis_coef_B),
                                         Rcpp::Named("I_A_sigma_sq", I_A_sigma_sq),
                                         Rcpp::Named("I_B_sigma_sq", I_B_sigma_sq),
                                         Rcpp::Named("LogLik", llik));
  return params;
}

inline Rcpp::List Total_sampler(const arma::field<arma::vec> X_A,
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
                                double& step_size,
                                double& step_size_delta,
                                const double& step_size_labels,
                                const int& num_evals,
                                const double prior_p_labels,
                                arma::mat& Mass_mat,
                                int Warm_block){
  arma::mat theta(MCMC_iters + Warm_block, init_position.n_elem, arma::fill::ones);
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec vec_accept(MCMC_iters + Warm_block, arma::fill::zeros);
  arma::vec vec_accept_delta(MCMC_iters + Warm_block, arma::fill::zeros);
  double prop_accept_10 = 0;
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  double llik = 0;
  int accept_num = 0;
  // Use initial starting position
  //Rcpp::Rcout << "Made it";
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
  arma::vec theta_exp;
 
  //Rcpp::Rcout << "Made it 2";
  for(int i = 1; i < Warm_block; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size =" << step_size << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
  
    theta_exp = arma::exp(theta_ph);
    llik = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    Rcpp::Rcout << llik;
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps,
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    //FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, prior_p_labels, accept_num);
    
    //sample_labels_step(Labels, i, X_AB, n_AB, theta_ph);
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    //Rcpp::Rcout << "Made it 4";
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
      
      // adjust step size for delta
      // prop_accept_10_delta = arma::accu(vec_accept_delta.subvec(i-9, i))/ 10;
      // if(prop_accept_10_delta  <= 0.1){
      //   step_size_delta = step_size_delta * 0.1;
      // }else if(prop_accept_10_delta <= 0.3){
      //   step_size_delta = step_size_delta * 0.5;
      // }else if(prop_accept_10_delta <= 0.6){
      //   step_size_delta = step_size_delta * 0.8;
      // }else if(prop_accept_10_delta > 0.85){
      //   step_size_delta = step_size_delta * 1.1;
      // }
    }
  }
  
  for(int i =  Warm_block; i <  Warm_block + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    //FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, prior_p_labels, accept_num);
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  Rcpp::Rcout << accept_num;
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", Labels));
  
  return params;
}


inline Rcpp::List Mixed_sampler(const arma::field<arma::vec> X_A,
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
                                double& step_size,
                                double& step_size_delta,
                                const double& step_size_labels,
                                const int& num_evals,
                                double delta_proposal_mean,
                                double delta_proposal_sd,
                                const double alpha,
                                int delta_adaption_block,
                                int M_proposal,
                                int n_Ensambler_sampler,
                                arma::mat& Mass_mat,
                                int Warm_block1,
                                int Warm_block2){
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, init_position.n_elem, arma::fill::ones);
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec l_lik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  //arma::vec vec_accept_delta(MCMC_iters + Warm_block, arma::fill::zeros);
  double prop_accept_10 = 0;
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block1 + Warm_block2);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  double llik = 0;
  int accept_num = 0;
  // Use initial starting position
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block1 + Warm_block2; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
  arma::vec theta_exp;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block 1, Iteration = " << i << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-25, i)) / 26 << "\n";
      //Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-25, i)) / 26 << "\n";
      //Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size = " << step_size << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps,
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    // Rcpp::Rcout << "delta = " << theta_ph(4);
    // FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    if((i % n_Ensambler_sampler) == 0){
      FFBS_ensemble_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                         num_evals, delta_proposal_mean, delta_proposal_sd, alpha,
                         M_proposal, delta_shape, delta_rate);
    }
    // Rcpp::Rcout << " delta_aft = " << theta_ph1(4) << "\n";
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    //Rcpp::Rcout << "Made it 4";
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
      
    }
  }
  double delta_proposal_meani = arma::mean(theta.col(4).subvec(Warm_block1 - delta_adaption_block, Warm_block1 - 1));
  double delta_proposal_sdi = arma::stddev(theta.col(4).subvec(Warm_block1 - delta_adaption_block, Warm_block1 - 1));
  // Rcpp::Rcout << "mean = " << delta_proposal_mean1 << " sd = " << delta_proposal_sd1;
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % delta_adaption_block) == 0){
      delta_proposal_meani = arma::mean(theta.col(4).subvec(i - delta_adaption_block, i - 1));
      delta_proposal_sdi = arma::stddev(theta.col(4).subvec(i - delta_adaption_block, i - 1));
      if(delta_proposal_sdi == 0.00){
        delta_proposal_sdi = 0.05;
      }
    }
    if((i % 50) == 0){
      Rcpp::Rcout << "Warm Up Block 2,  Iteration = " << i << "\n";
      //Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "(mu, sigma) = (" << delta_proposal_meani << ", " << delta_proposal_sdi << ") \n"; 
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n";
      //Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size = " << step_size << "\n" << "\n";

    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    //FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    // if((i % n_Ensambler_sampler) == 0){
    FFBS_ensemble_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                       num_evals, delta_proposal_meani, delta_proposal_sdi, alpha,
                       M_proposal, delta_shape, delta_rate);
    // }
    theta.row(i) = theta_ph.t();
    
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
      
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n" << "\n";

    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    // FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    // if((i % n_Ensambler_sampler) == 0){
    FFBS_ensemble_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                        num_evals, delta_proposal_meani, delta_proposal_sdi, alpha,
                        M_proposal, delta_shape, delta_rate);
    // }
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  Rcpp::Rcout << accept_num;
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", Labels));
  
  return params;
}

inline Rcpp::List Mixed_sampler_int(const arma::field<arma::vec> X_A,
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
                                    double& step_size,
                                    double& step_size_delta,
                                    const double& step_size_labels,
                                    const int& num_evals,
                                    double delta_proposal_mean,
                                    double delta_proposal_sd,
                                    const double alpha,
                                    int delta_adaption_block,
                                    int M_proposal,
                                    int n_Ensambler_sampler,
                                    arma::mat& Mass_mat,
                                    int Warm_block1,
                                    int Warm_block2){
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, init_position.n_elem, arma::fill::ones);
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec l_lik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  //arma::vec vec_accept_delta(MCMC_iters + Warm_block, arma::fill::zeros);
  double prop_accept_10 = 0;
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block1 + Warm_block2);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  double llik = 0;
  int accept_num = 0;
  // Use initial starting position
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block1 + Warm_block2; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
  arma::vec theta_exp;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "Warm Up Block 1, Iteration = " << i << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-25, i)) / 26 << "\n";
      //Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-25, i)) / 26 << "\n";
      //Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size = " << step_size << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps,
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    // Rcpp::Rcout << "delta = " << theta_ph(4);
    // FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    if((i % n_Ensambler_sampler) == 0){
      FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                         num_evals, delta_proposal_mean, delta_proposal_sd, alpha,
                         M_proposal, delta_shape, delta_rate);
    }
    // Rcpp::Rcout << " delta_aft = " << theta_ph1(4) << "\n";
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    //Rcpp::Rcout << "Made it 4";
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
      
    }
  }
  double delta_proposal_meani = arma::mean(theta.col(4).subvec(Warm_block1 - delta_adaption_block, Warm_block1 - 1));
  double delta_proposal_sdi = arma::stddev(theta.col(4).subvec(Warm_block1 - delta_adaption_block, Warm_block1 - 1));
  // Rcpp::Rcout << "mean = " << delta_proposal_mean1 << " sd = " << delta_proposal_sd1;
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % delta_adaption_block) == 0){
      delta_proposal_meani = arma::mean(theta.col(4).subvec(i - delta_adaption_block, i - 1));
      delta_proposal_sdi = arma::stddev(theta.col(4).subvec(i - delta_adaption_block, i - 1));
      if(delta_proposal_sdi == 0.00){
        delta_proposal_sdi = 0.05;
      }
    }
    if((i % 50) == 0){
      Rcpp::Rcout << "Warm Up Block 2,  Iteration = " << i << "\n";
      //Rcpp::Rcout << "Prob_accept (delta)= " << arma::accu(vec_accept_delta.subvec(i-25, i)) / 26 << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "(mu, sigma) = (" << delta_proposal_meani << ", " << delta_proposal_sdi << ") \n"; 
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n";
      //Rcpp::Rcout << "Step Size (delta)  = " << step_size_delta<< "\n";
      Rcpp::Rcout << "Step Size = " << step_size << "\n" << "\n";
      
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    //FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    // if((i % n_Ensambler_sampler) == 0){
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                        num_evals, delta_proposal_meani, delta_proposal_sdi, alpha,
                        M_proposal, delta_shape, delta_rate);
    // }
    theta.row(i) = theta_ph.t();
    
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.1){
        step_size = step_size * 0.1;
      }else if(prop_accept_10 <= 0.3){
        step_size = step_size * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size = step_size * 0.8;
      }else if(prop_accept_10 > 0.85){
        step_size = step_size * 1.5;
      }
      
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Log-likelihood = " << arma::accu(l_lik.subvec(i-50, i)) / 51 << "\n";
      Rcpp::Rcout << "Prob_accept = " << arma::accu(vec_accept.subvec(i-50, i)) / 51 << "\n" << "\n";
      
    }
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    theta_exp = arma::exp(theta_ph);
    l_lik(i) = log_likelihood(Labels_iter, theta_exp, X_A, X_B, X_AB, n_A, n_B, n_AB);
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps, 
             vec_accept(i));
    theta.row(i) = theta_ph.t();
    //FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, accept_num);
    // if((i % n_Ensambler_sampler) == 0){
      FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, step_size_labels,
                          num_evals, delta_proposal_meani, delta_proposal_sdi, alpha,
                          M_proposal, delta_shape, delta_rate);
    // }
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  Rcpp::Rcout << accept_num;
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", Labels));
  
  return params;
}
}


#endif