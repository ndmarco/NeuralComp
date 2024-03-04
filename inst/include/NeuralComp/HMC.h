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
                             double& step_size,
                             arma::mat& Mass_mat_inv,
                             arma::vec& position,
                             arma::vec& momentum,
                             ADFun<double>& gr,
                             int step_num,
                             int num_leapfrog){
  position.subvec(0, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1) = position.subvec(0, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1) + 
    step_size * Mass_mat_inv * momentum;
  arma::vec position_theta = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef_A = position.subvec(0, basis_funct_A(0,0).n_cols - 1);
  arma::vec position_basis_coef_B = position.subvec(basis_funct_A(0,0).n_cols, basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols - 1);
  //if(step_num != (num_leapfrog - 1)){
    momentum = momentum + step_size * 
      trans_calc_gradient_eigen_basis_update(Labels, position_theta, position_basis_coef_A, 
                                             position_basis_coef_B, basis_funct_A, basis_funct_B,
                                             basis_funct_AB, X_A, X_B, X_AB, n_A,
                                             n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, gr);
  //}
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
                                double& step_size,
                                arma::mat& Mass_mat_inv,
                                arma::vec& position,
                                arma::vec& momentum,
                                ADFun<double>& gr,
                                int step_num,
                                int num_leapfrog){
  // update position
  position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols,position.n_elem-2) = position.subvec(basis_funct_B(0,0).n_cols + basis_funct_A(0,0).n_cols,position.n_elem-2) + 
    step_size  * Mass_mat_inv * momentum;
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

inline void leapfrog_step_IGP_theta(const arma::field<arma::mat>& basis_funct,
                                    const arma::field<arma::vec>& X,
                                    const arma::vec& n,
                                    const double& I_mean, 
                                    const double& I_shape,
                                    const double& sigma_mean,
                                    const double& sigma_shape,
                                    double& step_size,
                                    arma::mat& Mass_mat_inv,
                                    arma::vec& position,
                                    arma::vec& momentum,
                                    ADFun<double>& gr,
                                    int step_num,
                                    int num_leapfrog){
  // update position
  position.subvec(basis_funct(0,0).n_cols, position.n_elem-1) = position.subvec(basis_funct(0,0).n_cols, position.n_elem-1) + 
    step_size  * Mass_mat_inv * momentum;
  arma::vec position_theta = position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = position.subvec(0, basis_funct(0,0).n_cols - 1);
  // update momentum
  //if(step_num != (num_leapfrog - 1)){
    momentum = momentum + step_size * 
      trans_calc_gradient_eigen_IGP_theta_update(position_theta, position_basis_coef, 
                                                 basis_funct, X, n, I_mean, I_shape,
                                                 sigma_mean, sigma_shape, gr);
  //}
  
}


inline void leapfrog_step_IGP_basis(const arma::field<arma::mat>& basis_funct,
                                    const arma::field<arma::vec>& X,
                                    const arma::vec& n,
                                    const double& I_sigma_sq,
                                    double& step_size,
                                    arma::mat& Mass_mat_inv,
                                    arma::vec& position,
                                    arma::vec& momentum,
                                    ADFun<double>& gr,
                                    int step_num,
                                    int num_leapfrog){
  // update position
  position.subvec(0, basis_funct(0,0).n_cols-1) = position.subvec(0, basis_funct(0,0).n_cols-1) + 
    step_size  * Mass_mat_inv * momentum;
  arma::vec position_theta = position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = position.subvec(0, basis_funct(0,0).n_cols - 1);
  // update momentum
  //if(step_num != (num_leapfrog - 1)){
    momentum = momentum + step_size * 
      trans_calc_gradient_eigen_IGP_basis_update(position_theta, position_basis_coef, 
                                                 basis_funct, X, n, I_sigma_sq, gr);
  //}
  
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
  arma::mat Mass_mat_inv = arma::inv_sympd(Mass_mat);
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_FR(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, 
                     X_B, X_AB, n_A, n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq,
                     step_size, Mass_mat_inv, prop_position, prop_momentum, gr, i, Leapfrog_steps);
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
  arma::mat Mass_mat_inv = arma::inv_sympd(Mass_mat);
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_theta(Labels, basis_funct_A, basis_funct_B, basis_funct_AB, X_A, 
                        X_B, X_AB, n_A, n_B, n_AB, I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        step_size, Mass_mat_inv, prop_position, prop_momentum, gr, i, Leapfrog_steps);
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

inline void leapfrog_IGP_basis(const arma::field<arma::mat>& basis_funct,
                               const arma::field<arma::vec>& X,
                               const arma::vec& n,
                               const double& I_sigma_sq,
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
  arma::vec position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  // Initial half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_IGP_basis(position_theta, position_basis_coef, basis_funct, 
                                        X, n, I_sigma_sq,  gr);
  arma::mat Mass_mat_inv = arma::inv_sympd(Mass_mat);
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_IGP_basis(basis_funct, X, n, I_sigma_sq, step_size, Mass_mat_inv, 
                            prop_position, prop_momentum, gr, i, Leapfrog_steps);
  }
  position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  // Final half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_IGP_basis_update(position_theta, position_basis_coef, 
                                               basis_funct, X, n, I_sigma_sq, gr);
}

inline void leapfrog_IGP_theta(const arma::field<arma::mat>& basis_funct,
                               const arma::field<arma::vec>& X,
                               const arma::vec& n,
                               const double& I_mean, 
                               const double& I_shape,
                               const double& sigma_mean,
                               const double& sigma_shape,
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
  arma::vec position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  // Initial half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_IGP_theta(position_theta, position_basis_coef, 
                                        basis_funct, X, n, I_mean, I_shape,
                                        sigma_mean, sigma_shape, gr);
  arma::mat Mass_mat_inv = arma::inv_sympd(Mass_mat);
  for(int i = 0; i < Leapfrog_steps; i++){
    leapfrog_step_IGP_theta(basis_funct, X, n, I_mean, I_shape, sigma_mean, sigma_shape,
                            step_size, Mass_mat_inv, prop_position, prop_momentum, gr, i, Leapfrog_steps);
  }
  
  position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  // Final half-step
  prop_momentum = prop_momentum + 0.5 * step_size * 
    trans_calc_gradient_eigen_IGP_theta_update(position_theta, position_basis_coef, 
                                               basis_funct, X, n, I_mean, I_shape,
                                               sigma_mean, sigma_shape, gr);
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

inline double lprob_accept_IGP_basis(arma::vec& prop_position,
                                     arma::vec& prop_momentum,
                                     arma::vec& position,
                                     arma::vec& momentum,
                                     const arma::field<arma::mat>& basis_funct,
                                     const arma::field<arma::vec>& X,
                                     const arma::vec& n,
                                     const double& I_sigma_sq,
                                     arma::mat& Mass_mat){
  arma::vec position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  
  double lp_accept = transformed_log_posterior_IGP_basis(position_theta, position_basis_coef,
                                                         basis_funct, X, n, I_sigma_sq);
  
  lp_accept = lp_accept - 0.5 * arma::dot(arma::solve(Mass_mat, prop_momentum), prop_momentum);
  
  position_theta = position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  position_basis_coef = position.subvec(0, basis_funct(0,0).n_cols - 1);
  
  lp_accept = lp_accept - transformed_log_posterior_IGP_basis(position_theta, position_basis_coef,
                                                              basis_funct, X, n, I_sigma_sq);
  lp_accept = lp_accept + 0.5 * arma::dot(arma::solve(Mass_mat, momentum), momentum);
  return lp_accept;
}

inline double lprob_accept_IGP_theta(arma::vec& prop_position,
                                     arma::vec& prop_momentum,
                                     arma::vec& position,
                                     arma::vec& momentum,
                                     const arma::field<arma::mat>& basis_funct,
                                     const arma::field<arma::vec>& X,
                                     const arma::vec& n,
                                     const double& I_mean, 
                                     const double& I_shape,
                                     const double& sigma_mean,
                                     const double& sigma_shape,
                                     arma::mat& Mass_mat){
  arma::vec position_theta = prop_position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  arma::vec position_basis_coef = prop_position.subvec(0, basis_funct(0,0).n_cols - 1);
  
  double lp_accept = transformed_log_posterior_IGP_theta(position_theta, position_basis_coef,
                                                         basis_funct, X, n, I_mean, I_shape,
                                                         sigma_mean, sigma_shape);
  
  lp_accept = lp_accept - 0.5 * arma::dot(arma::solve(Mass_mat, prop_momentum), prop_momentum);
  
  position_theta = position.subvec(basis_funct(0,0).n_cols, position.n_elem - 1);
  position_basis_coef = position.subvec(0, basis_funct(0,0).n_cols - 1);
  
  lp_accept = lp_accept - transformed_log_posterior_IGP_theta(position_theta, position_basis_coef,
                                                              basis_funct, X, n, I_mean, I_shape, 
                                                              sigma_mean, sigma_shape);
  lp_accept = lp_accept + 0.5 * arma::dot(arma::solve(Mass_mat, momentum), momentum);
  return lp_accept;
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
              n_A, n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq, step_size, Mass_mat,
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
                 step_size, Mass_mat, position, momentum, prop_position,
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

inline void HMC_step_IGP_basis(arma::vec& theta,
                               arma::vec& basis_coef,
                               const arma::field<arma::mat>& basis_funct,
                               const arma::field<arma::vec>& X,
                               const arma::vec& n,
                               const double& I_sigma_sq,
                               double& step_size,
                               arma::mat& Mass_mat,
                               int Leapfrog_steps,
                               double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = arma::mvnrnd(arma::zeros(basis_coef.n_elem), Mass_mat);
  arma::vec prop_position = arma::zeros(basis_coef.n_elem + theta.n_elem);
  prop_position.subvec(0, basis_coef.n_elem -1) = basis_coef;
  prop_position.subvec(basis_coef.n_elem, prop_position.n_elem - 1) = theta;
  arma::vec position = prop_position;
  arma::vec prop_momentum = momentum;
  leapfrog_IGP_basis(basis_funct, X, n, I_sigma_sq, step_size, Mass_mat,
                     position, momentum, prop_position, prop_momentum, Leapfrog_steps);
  double accept = lprob_accept_IGP_basis(prop_position, prop_momentum, position, momentum,
                                         basis_funct, X, n, I_sigma_sq, Mass_mat);
  
  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    basis_coef = prop_position.subvec(0, basis_coef.n_elem -1);
  }
  
}



inline void HMC_step_IGP_theta(arma::vec& theta,
                               arma::vec& basis_coef,
                               const arma::field<arma::mat>& basis_funct,
                               const arma::field<arma::vec>& X,
                               const arma::vec& n,
                               const double& I_mean, 
                               const double& I_shape,
                               const double& sigma_mean,
                               const double& sigma_shape,
                               double& step_size,
                               arma::mat& Mass_mat,
                               int Leapfrog_steps,
                               double& num_accept){
  // Sample for I_A, I_B, sigma_A, sigma_B
  arma::vec momentum = arma::mvnrnd(arma::zeros(theta.n_elem), Mass_mat);
  arma::vec prop_position = arma::zeros(basis_coef.n_elem + theta.n_elem);
  prop_position.subvec(0, basis_coef.n_elem -1) = basis_coef;
  prop_position.subvec(basis_coef.n_elem, prop_position.n_elem - 1) = theta;
  arma::vec position = prop_position;
  arma::vec prop_momentum = momentum;
  leapfrog_IGP_theta(basis_funct, X, n, I_mean, I_shape, sigma_mean, sigma_shape,
                     step_size, Mass_mat, position, momentum, prop_position,
                     prop_momentum, Leapfrog_steps);
  
  double accept = lprob_accept_IGP_theta(prop_position, prop_momentum, position, momentum,
                                         basis_funct, X, n, I_mean, I_shape, sigma_mean,
                                         sigma_shape, Mass_mat);
  if(std::log(R::runif(0,1)) < accept){
    num_accept = 1;
    theta = prop_position.subvec(basis_coef.n_elem, prop_position.n_elem - 1);
  }
  
}


inline Rcpp::List Mixed_sampler_int(const arma::field<arma::vec> X_A,
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
                                    const double delta_shape,
                                    const double delta_rate,
                                    double& step_size_theta,
                                    double delta_proposal_mean,
                                    double delta_proposal_sd,
                                    const double alpha_labels,
                                    int delta_adaption_block,
                                    int theta_adaptation_block,
                                    int M_proposal,
                                    int Warm_block1,
                                    int Warm_block2){
  arma::field<arma::mat> basis_funct_A(n_A.n_elem, 1);
  for(int i = 0; i < n_A.n_elem; i++){
    basis_funct_A(i,0) = arma::zeros(n_A(i), 1);
  }
  arma::field<arma::mat> basis_funct_B(n_B.n_elem, 1);
  for(int i = 0; i < n_B.n_elem; i++){
    basis_funct_B(i,0) = arma::zeros(n_B(i), 1);
  }
  arma::field<arma::mat> basis_funct_AB(n_AB.n_elem, 1);
  for(int i = 0; i < n_AB.n_elem; i++){
    basis_funct_AB(i,0) = arma::zeros(n_AB(i), 1);
  }
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, 5, arma::fill::ones);
  arma::mat basis_coef_A(MCMC_iters + Warm_block1 + Warm_block2, 1, arma::fill::zeros);
  arma::mat basis_coef_B(MCMC_iters + Warm_block1 + Warm_block2, 1, arma::fill::zeros);
  arma::vec I_A_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec I_B_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec llik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec lposterior(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec basis_coef_A_ph = basis_coef_A.row(0).t();
  arma::vec basis_coef_B_ph = basis_coef_B.row(0).t();
  
  arma::vec init_position(5, arma::fill::ones);
  init_position(0) = I_A_mean;
  init_position(1) = I_B_mean;
  init_position(2) = sigma_A_mean;
  init_position(3) = sigma_B_mean;
  init_position(4) = delta_shape / delta_rate;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block1 + Warm_block2);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  
  // Use initial starting position
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block1 + Warm_block2; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
  arma::vec theta_exp;
  
  
  arma::vec vec_accept_FR(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec vec_accept_theta(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::mat Mass_mat_theta = arma::diagmat(arma::ones(theta.n_cols-1));
  arma::mat Mass_mat_basis = arma::diagmat(arma::ones(basis_coef_A.n_cols + basis_coef_B.n_cols));
  double prop_accept_10 = 0;
  double prop_accept_10_theta = 0;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 1) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_mean, delta_proposal_sd, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    lposterior(i) = log_posterior_model(llik(i), theta_exp, I_A_mean, I_A_shape, 
               I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    //Rcpp::Rcout << "Made it 4";
    if((i % 10) == 0){
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_theta<= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.85){
        step_size_theta = step_size_theta * 2;
      }
    }
    
  }
  
  double delta_proposal_meani = arma::mean(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  double delta_proposal_sdi = arma::stddev(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  if(delta_proposal_sdi == 0.00){
    delta_proposal_sdi = 0.005;
  }
  
  Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(Warm_block1 - std::floor(0.5 *Warm_block1),0, Warm_block1 -1, theta.n_cols - 2)));
  
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 2) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "(mu, sigma) = (" << delta_proposal_meani << ", " << delta_proposal_sdi << ") \n"; 
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
    }
    
    if(i > Warm_block1){
      if(((i - Warm_block1) % delta_adaption_block) == 0){
        delta_proposal_meani = arma::mean(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        delta_proposal_sdi = arma::stddev(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        if(delta_proposal_sdi == 0.00){
          delta_proposal_sdi = 0.005;
        }
      }
      if(((i - Warm_block1) % theta_adaptation_block) == 0){
        Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(i - theta_adaptation_block,0, i-1, theta.n_cols - 2)));
      }
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_meani, delta_proposal_sdi, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    lposterior(i) = log_posterior_model(llik(i), theta_exp, I_A_mean, I_A_shape, 
               I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    
    // set labels for current MCMC iteration
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.8){
        step_size_theta = step_size_theta * 1.2;
      }else if(prop_accept_10_theta > 0.9){
        step_size_theta = step_size_theta * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-50, i-1)) / 50 << "\n";
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_meani, delta_proposal_sdi, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    lposterior(i) = log_posterior_model(llik(i), theta_exp, I_A_mean, I_A_shape, 
               I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate);
    
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  //convert labels
  arma::field<arma::mat> labels_out(1, n_AB.n_elem);
  for(int i = 0; i < n_AB.n_elem; i++){
    labels_out(0,i) = arma::zeros(Warm_block1 + Warm_block2 + MCMC_iters, n_AB(i));
    for(int j = 0; j < Warm_block1 + Warm_block2 + MCMC_iters; j++){
      for(int k = 0; k < n_AB(i); k++){
        labels_out(0,i)(j,k) = Labels(i,j)(k);
      }
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", labels_out),
                                         Rcpp::Named("LogLik", llik),
                                         Rcpp::Named("LogPosterior", lposterior));
  
  return params;
}



inline Rcpp::List Mixed_sampler_int_TI(const arma::field<arma::mat>& basis_funct_A,
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
                                       const double delta_shape,
                                       const double delta_rate,
                                       double& step_size_theta,
                                       double& step_size_FR,
                                       double delta_proposal_mean,
                                       double delta_proposal_sd,
                                       const double alpha_labels,
                                       const double gamma,
                                       int delta_adaption_block,
                                       int theta_adaptation_block,
                                       int M_proposal,
                                       int Warm_block1,
                                       int Warm_block2){
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, 5, arma::fill::ones);
  arma::mat basis_coef_A(MCMC_iters + Warm_block1 + Warm_block2, basis_funct_A(0,0).n_cols, arma::fill::zeros);
  arma::mat basis_coef_B(MCMC_iters + Warm_block1 + Warm_block2, basis_funct_B(0,0).n_cols, arma::fill::zeros);
  arma::vec I_A_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec omega_A(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec I_B_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec omega_B(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec llik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec lposterior(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec basis_coef_A_ph = basis_coef_A.row(0).t();
  arma::vec basis_coef_B_ph = basis_coef_B.row(0).t();
  
  arma::vec init_position(5, arma::fill::ones);
  init_position(0) = I_A_mean;
  init_position(1) = I_B_mean;
  init_position(2) = sigma_A_mean;
  init_position(3) = sigma_B_mean;
  init_position(4) = delta_shape / delta_rate;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block1 + Warm_block2);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  
  // Use initial starting position
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block1 + Warm_block2; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
  arma::vec theta_exp;
  
  
  arma::vec vec_accept_FR(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec vec_accept_theta(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::mat Mass_mat_theta = arma::diagmat(arma::ones(theta.n_cols-1));
  arma::mat Mass_mat_basis = arma::diagmat(arma::ones(basis_coef_A.n_cols + basis_coef_B.n_cols));
  double prop_accept_10 = 0;
  double prop_accept_10_theta = 0;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 1) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR = " << arma::accu(vec_accept_FR.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR = " << step_size_FR << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_mean, delta_proposal_sd, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    HMC_step_FR(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB,
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma_cauchy(basis_coef_A_ph, i, omega_A, I_A_sigma_sq);
    update_omega(gamma, i, omega_A, I_A_sigma_sq);
    update_I_sigma_cauchy(basis_coef_B_ph, i, omega_B, I_B_sigma_sq);
    update_omega(gamma, i, omega_B, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    // lposterior(i) = log_posterior_model_TI(llik(i), theta_exp, basis_coef_A_ph, basis_coef_B_ph,
    //            I_A_sigma_sq(i), I_B_sigma_sq(i), I_A_mean, I_A_shape,
    //            I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
    //            sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
    //            alpha, beta);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
      omega_A(i + 1) = omega_A(i);
      omega_B(i + 1) = omega_B(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
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
      }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_theta<= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.85){
        step_size_theta = step_size_theta * 2;
      }
    }
    
  }
  
  double delta_proposal_meani = arma::mean(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  double delta_proposal_sdi = arma::stddev(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  if(delta_proposal_sdi == 0.00){
    delta_proposal_sdi = 0.005;
  }
  
  Mass_mat_theta = arma::inv_sympd(arma::diagmat(arma::diagvec(arma::cov(theta.submat(Warm_block1 - std::floor(0.5 *Warm_block1),0, Warm_block1 -1, theta.n_cols - 2)))));
  arma::mat ph_basis = arma::zeros(std::ceil(0.5 *Warm_block1), basis_coef_A.n_cols + basis_coef_B.n_cols);
  ph_basis.submat(0, 0, std::ceil(0.5 *Warm_block1) - 1, basis_coef_A.n_cols-1) = basis_coef_A.submat(Warm_block1 - std::floor(0.5 *Warm_block1), 0,
                  Warm_block1 - 1, basis_coef_A.n_cols - 1);
  ph_basis.submat(0, basis_coef_A.n_cols, std::ceil(0.5 *Warm_block1) - 1, basis_coef_A.n_cols + basis_coef_B.n_cols - 1) = basis_coef_B.submat(Warm_block1 - std::floor(0.5 *Warm_block1), 0,
                  Warm_block1 - 1, basis_coef_B.n_cols - 1);
  Mass_mat_basis = arma::inv_sympd(arma::diagmat(arma::diagvec(arma::cov(ph_basis))));
  
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 2) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR= " << arma::accu(vec_accept_FR.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "(mu, sigma) = (" << delta_proposal_meani << ", " << delta_proposal_sdi << ") \n"; 
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR = " << step_size_FR << "\n" << "\n";
    }
    
    if(i > Warm_block1){
      if(((i - Warm_block1) % delta_adaption_block) == 0){
        delta_proposal_meani = arma::mean(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        delta_proposal_sdi = arma::stddev(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        if(delta_proposal_sdi == 0.00){
          delta_proposal_sdi = 0.005;
        }
      }
      if(((i - Warm_block1) % theta_adaptation_block) == 0){
        Mass_mat_theta = arma::inv_sympd(arma::diagmat(arma::diagvec(arma::cov(theta.submat(i - theta_adaptation_block, 0, i-1, theta.n_cols - 2)))));
        arma::mat ph_basis1 = arma::zeros(theta_adaptation_block, basis_coef_A.n_cols + basis_coef_B.n_cols);
        ph_basis1.submat(0, 0, theta_adaptation_block - 1, basis_coef_A.n_cols-1) = basis_coef_A.submat(i- theta_adaptation_block, 0,
                         i - 1, basis_coef_A.n_cols - 1);
        ph_basis1.submat(0, basis_coef_A.n_cols, theta_adaptation_block - 1, basis_coef_A.n_cols + basis_coef_B.n_cols - 1) = basis_coef_B.submat(i- theta_adaptation_block, 0,
                         i - 1, basis_coef_B.n_cols - 1);
        Mass_mat_basis = arma::inv_sympd(arma::diagmat(arma::diagvec(arma::cov(ph_basis1))));
      }
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_meani, delta_proposal_sdi, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    HMC_step_FR(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB,
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma_cauchy(basis_coef_A_ph, i, omega_A, I_A_sigma_sq);
    update_omega(gamma, i, omega_A, I_A_sigma_sq);
    update_I_sigma_cauchy(basis_coef_B_ph, i, omega_B, I_B_sigma_sq);
    update_omega(gamma, i, omega_B, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);

    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    
    // set labels for current MCMC iteration
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
      omega_A(i + 1) = omega_A(i);
      omega_B(i + 1) = omega_B(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept_FR.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.3){
        step_size_FR = step_size_FR * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size_FR = step_size_FR * 0.8;
      }else if(prop_accept_10 > 0.8){
        step_size_FR = step_size_FR * 1.2;
      }else if(prop_accept_10 > 0.9){
        step_size_FR = step_size_FR * 1.5;
      }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.8){
        step_size_theta = step_size_theta * 1.2;
      }else if(prop_accept_10_theta > 0.9){
        step_size_theta = step_size_theta * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log joint prob = " << arma::accu(llik.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept FR = " << arma::accu(vec_accept_FR.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-50, i-1)) / 50 << "\n";
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_A_ph = basis_coef_A.row(i).t();
    basis_coef_B_ph = basis_coef_B.row(i).t();
    theta_ph = theta.row(i).t();
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A_ph, 
                        basis_coef_B_ph, basis_funct_AB,
                        delta_proposal_meani, delta_proposal_sdi, alpha_labels,
                        M_proposal, delta_shape, delta_rate);
    
    // set labels for current MCMC iteration
    for(int j = 0; j < n_AB.n_elem; j++){
      Labels_iter(j,0) = Labels(j, i);
    }
    
    HMC_step_theta(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                   basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB, 
                   I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                   sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                   step_size_theta, Mass_mat_theta, Leapfrog_steps, vec_accept_theta(i));
    HMC_step_FR(Labels_iter, theta_ph, basis_coef_A_ph, basis_coef_B_ph, basis_funct_A,
                basis_funct_B, basis_funct_AB, X_A, X_B, X_AB, n_A, n_B, n_AB,
                I_A_sigma_sq(i), I_B_sigma_sq(i),
                step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma_cauchy(basis_coef_A_ph, i, omega_A, I_A_sigma_sq);
    update_omega(gamma, i, omega_A, I_A_sigma_sq);
    update_I_sigma_cauchy(basis_coef_B_ph, i, omega_B, I_B_sigma_sq);
    update_omega(gamma, i, omega_B, I_B_sigma_sq);
    
    llik(i) = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A_ph, basis_coef_B_ph,
         basis_funct_A, basis_funct_B, basis_funct_AB,
         X_A, X_B, X_AB, n_A, n_B, n_AB);
    // lposterior(i) = log_posterior_model_TI(llik(i), theta_exp, basis_coef_A_ph, basis_coef_B_ph,
    //            I_A_sigma_sq(i), I_B_sigma_sq(i), I_A_mean, I_A_shape,
    //            I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
    //            sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
    //            alpha, beta);
    
    theta.row(i) = theta_ph.t();
    basis_coef_A.row(i) = basis_coef_A_ph.t();
    basis_coef_B.row(i) = basis_coef_B_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef_A.row(i + 1) = basis_coef_A.row(i);
      basis_coef_B.row(i + 1) = basis_coef_B.row(i);
      I_A_sigma_sq(i + 1) = I_A_sigma_sq(i);
      I_B_sigma_sq(i + 1) = I_B_sigma_sq(i);
      omega_A(i + 1) = omega_A(i);
      omega_B(i + 1) = omega_B(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  
  //convert labels
  arma::field<arma::mat> labels_out(n_AB.n_elem, 1);
  for(int i = 0; i < n_AB.n_elem; i++){
    labels_out(i,0) = arma::zeros(Warm_block1 + Warm_block2 + MCMC_iters, n_AB(i));
    for(int j = 0; j < Warm_block1 + Warm_block2 + MCMC_iters; j++){
      for(int k = 0; k < n_AB(i); k++){
        labels_out(i,0)(j,k) = Labels(i,j)(k);
      }
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", labels_out),
                                         Rcpp::Named("basis_coef_A", basis_coef_A),
                                         Rcpp::Named("basis_coef_B", basis_coef_B),
                                         Rcpp::Named("I_A_sigma_sq", I_A_sigma_sq),
                                         Rcpp::Named("I_B_sigma_sq", I_B_sigma_sq),
                                         Rcpp::Named("omega_A", omega_A),
                                         Rcpp::Named("omega_B", omega_B),
                                         Rcpp::Named("Mass_mat_theta", Mass_mat_theta),
                                         Rcpp::Named("Mass_mat_bsis", Mass_mat_basis),
                                         Rcpp::Named("LogLik", llik),
                                         Rcpp::Named("LogPosterior", lposterior));
  
  return params;
}


inline Rcpp::List Mixed_sampler_IGP_int(const arma::field<arma::vec> X,
                                        const arma::vec n,
                                        int MCMC_iters,
                                        int Leapfrog_steps,
                                        const double& I_mean, 
                                        const double& I_shape,
                                        const double sigma_mean,
                                        const double sigma_shape,
                                        double& step_size_theta,
                                        int theta_adaptation_block,
                                        int Warm_block1,
                                        int Warm_block2){
  arma::field<arma::mat> basis_funct(n.n_elem, 1);
  for(int i = 0; i < n.n_elem; i++){
    basis_funct(i,0) = arma::zeros(n(i), 1);
  }
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, 2, arma::fill::ones);
  arma::mat basis_coef(MCMC_iters + Warm_block1 + Warm_block2, 1, arma::fill::zeros);
  arma::vec I_A_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec I_B_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec llik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec lposterior(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec basis_coef_ph = basis_coef.row(0).t();
  
  arma::vec init_position(2, arma::fill::ones);
  init_position(0) = I_mean;
  init_position(1) = sigma_mean;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  
  arma::vec theta_exp;
  
  arma::vec vec_accept_theta(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::mat Mass_mat_theta = arma::diagmat(arma::ones(theta.n_cols));
  arma::mat Mass_mat_basis = arma::diagmat(arma::ones(basis_coef.n_cols));
  double prop_accept_10 = 0;
  double prop_accept_10_theta = 0;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 1) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model(llik(i), theta_exp, I_mean, I_shape, 
               sigma_mean, sigma_shape);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
    }

    if((i % 10) == 0){
      // adjust step size for I_A and sigma
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_theta<= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.5){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.85){
        step_size_theta = step_size_theta * 2;
      }
    }
    
  }
  
  
  
  Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(Warm_block1 - std::floor(0.5 *Warm_block1),0, Warm_block1 -1, theta.n_cols - 1)));
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 2) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
    }
    
    if(i > Warm_block1){
      if(((i - Warm_block1) % theta_adaptation_block) == 0){
        Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(i - theta_adaptation_block,0, i-1, theta.n_cols - 1)));
      }
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model(llik(i), theta_exp, I_mean, I_shape, 
               sigma_mean, sigma_shape);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
    }
    
    if((i % 10) == 0){
      // adjust step size for I and sigma
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.5){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.8){
        step_size_theta = step_size_theta * 1.2;
      }else if(prop_accept_10_theta > 0.9){
        step_size_theta = step_size_theta * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-50, i-1)) / 50 << "\n";
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    theta_exp = arma::exp(theta_ph);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model(llik(i), theta_exp, I_mean, I_shape, 
               sigma_mean, sigma_shape);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("LogLik", llik),
                                         Rcpp::Named("LogPosterior", lposterior));
  
  return params;
}

inline Rcpp::List Mixed_sampler_IGP_int_TI(const arma::field<arma::mat>& basis_funct,
                                           const arma::field<arma::vec> X,
                                           const arma::vec n,
                                           int MCMC_iters,
                                           int Leapfrog_steps,
                                           const double& I_mean, 
                                           const double& I_shape,
                                           const double sigma_mean,
                                           const double sigma_shape,
                                           double& step_size_theta,
                                           double& step_size_FR,
                                           const double alpha,
                                           const double beta,
                                           int theta_adaptation_block,
                                           int Warm_block1,
                                           int Warm_block2){
  arma::mat theta(MCMC_iters + Warm_block1 + Warm_block2, 2, arma::fill::ones);
  arma::mat basis_coef(MCMC_iters + Warm_block1 + Warm_block2, basis_funct(0,0).n_cols, arma::fill::zeros);
  arma::vec I_sigma_sq(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::ones);
  arma::vec vec_accept(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec llik(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec lposterior(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec basis_coef_ph = basis_coef.row(0).t();
  
  arma::vec init_position(2, arma::fill::ones);
  init_position(0) = I_mean;
  init_position(1) = sigma_mean;
  theta.row(0) = arma::log(init_position.t());
  theta.row(1) = arma::log(init_position.t());
  arma::vec theta_ph(init_position.n_elem);
  
  arma::vec theta_exp;
  
  arma::vec vec_accept_FR(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::vec vec_accept_theta(MCMC_iters + Warm_block1 + Warm_block2, arma::fill::zeros);
  arma::mat Mass_mat_theta = arma::diagmat(arma::ones(theta.n_cols));
  arma::mat Mass_mat_basis = arma::diagmat(arma::ones(basis_coef.n_cols));
  double prop_accept_10 = 0;
  double prop_accept_10_theta = 0;
  
  for(int i = 1; i < Warm_block1; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 1) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR = " << arma::accu(vec_accept_FR.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR = " << step_size_FR << "\n" << "\n";
    }
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    
    HMC_step_IGP_basis(theta_ph, basis_coef_ph, basis_funct, X, n, I_sigma_sq(i),
                       step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_ph, 0, alpha, beta, i, I_sigma_sq);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model_TI(llik(i), theta_exp, basis_coef_ph,
               I_sigma_sq(i), I_mean, I_shape, 
               sigma_mean, sigma_shape, alpha, beta);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef.row(i + 1) = basis_coef.row(i);
      I_sigma_sq(i + 1) = I_sigma_sq(i);
    }
    //Rcpp::Rcout << "Made it 4";
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
      }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.1){
        step_size_theta = step_size_theta * 0.1;
      }else if(prop_accept_10_theta<= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.85){
        step_size_theta = step_size_theta * 2;
      }
    }
    
  }
  
  Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(Warm_block1 - std::floor(0.5 *Warm_block1),0, Warm_block1 - 1, theta.n_cols - 1)));
  arma::mat ph_basis = arma::zeros(std::ceil(0.5 *Warm_block1), basis_coef.n_cols);
  ph_basis.submat(0, 0, std::ceil(0.5 *Warm_block1) - 1, basis_coef.n_cols-1) = basis_coef.submat(Warm_block1 - std::floor(0.5 *Warm_block1), 0, 
                  Warm_block1 - 1, basis_coef.n_cols - 1);
  Mass_mat_basis = arma::inv_sympd(arma::cov(ph_basis));
  
  
  for(int i =  Warm_block1; i <  Warm_block1 + Warm_block2; i++){
    if((i % 25) == 0){
      Rcpp::Rcout << "(Warm Up Block 2) Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept FR= " << arma::accu(vec_accept_FR.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-25, i-1)) / 25 << "\n";
      Rcpp::Rcout << "Step Size theta = " << step_size_theta << "\n";
      Rcpp::Rcout << "Step Size FR = " << step_size_FR << "\n" << "\n";
    }
    
    if(i > Warm_block1){
      if(((i - Warm_block1) % theta_adaptation_block) == 0){
        Mass_mat_theta = arma::inv_sympd(arma::cov(theta.submat(i - theta_adaptation_block,0, i-1, theta.n_cols - 1)));
        arma::mat ph_basis1 = arma::zeros(theta_adaptation_block, basis_coef.n_cols);
        ph_basis1.submat(0, 0, theta_adaptation_block - 1, basis_coef.n_cols-1) = basis_coef.submat(i- theta_adaptation_block, 0, 
                         i - 1, basis_coef.n_cols - 1);
        Mass_mat_basis = arma::inv_sympd(arma::cov(ph_basis1));
      }
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    
    HMC_step_IGP_basis(theta_ph, basis_coef_ph, basis_funct, X, n, I_sigma_sq(i),
                       step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_ph, 0, alpha, beta, i, I_sigma_sq);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model_TI(llik(i), theta_exp, basis_coef_ph,
               I_sigma_sq(i), I_mean, I_shape, 
               sigma_mean, sigma_shape, alpha, beta);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef.row(i + 1) = basis_coef.row(i);
      I_sigma_sq(i + 1) = I_sigma_sq(i);
    }
    
    if((i % 10) == 0){
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10 = arma::accu(vec_accept_FR.subvec(i-9, i))/ 10;
      if(prop_accept_10  <= 0.3){
        step_size_FR = step_size_FR * 0.5;
      }else if(prop_accept_10 <= 0.6){
        step_size_FR = step_size_FR * 0.8;
      }else if(prop_accept_10 > 0.8){
        step_size_FR = step_size_FR * 1.2;
      }else if(prop_accept_10 > 0.9){
        step_size_FR = step_size_FR * 1.5;
      }
      
      // adjust step size for I_A, I_B, sigma_A, sigma_B
      prop_accept_10_theta = arma::accu(vec_accept_theta.subvec(i-9, i))/ 10;
      if(prop_accept_10_theta  <= 0.3){
        step_size_theta = step_size_theta * 0.5;
      }else if(prop_accept_10_theta <= 0.6){
        step_size_theta = step_size_theta * 0.8;
      }else if(prop_accept_10_theta > 0.8){
        step_size_theta = step_size_theta * 1.2;
      }else if(prop_accept_10_theta > 0.9){
        step_size_theta = step_size_theta * 1.5;
      }
    }
  }
  
  for(int i =  Warm_block1 + Warm_block2; i <  Warm_block1 + Warm_block2 + MCMC_iters; i++){
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration = " << i << "\n";
      Rcpp::Rcout << "Avg log likelihood = " << arma::accu(llik.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept FR = " << arma::accu(vec_accept_FR.subvec(i-50, i-1)) / 50 << "\n";
      Rcpp::Rcout << "Prob_accept theta = " << arma::accu(vec_accept_theta.subvec(i-50, i-1)) / 50 << "\n";
    }
    
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    
    theta_ph = theta.row(i).t();
    basis_coef_ph = basis_coef.row(i).t();
    
    HMC_step_IGP_theta(theta_ph, basis_coef_ph, basis_funct, X, n, I_mean, I_shape,
                       sigma_mean, sigma_shape, step_size_theta, Mass_mat_theta, 
                       Leapfrog_steps, vec_accept_theta(i));
    
    HMC_step_IGP_basis(theta_ph, basis_coef_ph, basis_funct, X, n, I_sigma_sq(i),
                       step_size_FR, Mass_mat_basis, Leapfrog_steps, vec_accept_FR(i));
    theta_exp = arma::exp(theta_ph);
    
    // update sigma hyperparameters
    update_I_sigma(basis_coef_ph, 0, alpha, beta, i, I_sigma_sq);
    
    llik(i) = log_likelihood_IGP_theta(theta_exp, basis_coef_ph, basis_funct, X, n);
    lposterior(i) = log_posterior_IGP_model_TI(llik(i), theta_exp, basis_coef_ph,
               I_sigma_sq(i), I_mean, I_shape, 
               sigma_mean, sigma_shape, alpha, beta);
    
    theta.row(i) = theta_ph.t();
    basis_coef.row(i) = basis_coef_ph.t();
    if((i+1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      basis_coef.row(i + 1) = basis_coef.row(i);
      I_sigma_sq(i + 1) = I_sigma_sq(i);
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("basis_coef", basis_coef),
                                         Rcpp::Named("I_sigma_sq", I_sigma_sq),
                                         Rcpp::Named("LogLik", llik),
                                         Rcpp::Named("LogPosterior", lposterior));
  
  return params;
}

}


#endif



