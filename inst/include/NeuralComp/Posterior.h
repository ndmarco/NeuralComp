#ifndef NeuralComp_Posterior_H
#define NeuralComp_Posterior_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include <splines2Armadillo.h>
#include <CppAD.h>
using namespace CppAD;
using namespace Eigen;

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
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_A(i,0).row(j), basis_coef_A)))),
                                               pow((1 / theta(2)), 2));
    }
  }
  
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_B(i,0).row(j), basis_coef_B)))),
                                                   pow((1 / theta(3)), 2));
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
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))),
                                                     pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in B)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
          }else{
            // Condition if spike has switched from A to B
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
        }
      }
    }
  }
  return l_likelihood;
}

typedef AD<double> a_double;
typedef Matrix<a_double, Eigen::Dynamic, 1> a_vector;


inline a_double log_likelihood_eigen_theta(arma::field<arma::vec>& Labels,
                                           a_vector theta,
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

  a_double l_likelihood = 0;
  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_A(i,0).row(j), basis_coef_A)))),
                                               pow((1 / theta(2)), 2));
    }
  }
  
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_B(i,0).row(j), basis_coef_B)))),
                                               pow((1 / theta(3)), 2));
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
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))),
                                                     pow((1 / theta(3)), 2)) +
                                                       dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in B)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
          }else{
            // Condition if spike has switched from A to B
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_A)))), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB(i,0).row(j), basis_coef_B)))), pow((1 / theta(3)), 2));
        }
      }
    }
  }
  return l_likelihood;
}



inline a_double log_likelihood_eigen_basis(arma::field<arma::vec>& Labels,
                                           arma::vec& theta,
                                           a_vector basis_coef,
                                           const arma::field<arma::mat>& basis_funct_A,
                                           const arma::field<arma::mat>& basis_funct_B,
                                           const arma::field<arma::mat>& basis_funct_AB,
                                           const arma::field<arma::vec>& X_A,
                                           const arma::field<arma::vec>& X_B,
                                           const arma::field<arma::vec>& X_AB,
                                           const arma::vec& n_A,
                                           const arma::vec& n_B,
                                           const arma::vec& n_AB){
  
  a_double l_likelihood = 0;
  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_A(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_A(i,0).row(j), basis_coef, true)))),
                                               pow((1 / theta(2)), 2));
    }
  }
  
  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X_B(i,0)(j), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_B(i,0).row(j), basis_coef, false)))),
                                               pow((1 / theta(3)), 2));
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
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))),
                                                     pow((1 / theta(3)), 2)) +
                                                       dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))), pow((1 / theta(3)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in B)
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))), pow((1 / theta(3)), 2));
          }else{
            // Condition if spike has switched from A to B
            l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2)) +
              dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))), pow((1 / theta(3)), 2));
          }
        }else{
          l_likelihood = l_likelihood + pinv_gauss(X_AB(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, true)))), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(i,0)(j), (1 / (theta(1) * CppAD::exp(dot_AD(basis_funct_AB(i,0).row(j), basis_coef, false)))), pow((1 / theta(3)), 2));
        }
      }
    }
  }
  return l_likelihood;
}


inline double log_likelihood_IGP_theta(arma::vec theta,
                                       arma::vec basis_coef,
                                       const arma::field<arma::mat>& basis_funct,
                                       const arma::field<arma::vec>& X,
                                       const arma::vec& n){
  double l_likelihood = 0;
  // Calculate log-likelihood 
  for(int i = 0; i < n.n_elem; i++){
    for(int j = 0; j < n(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct(i,0).row(j), basis_coef)))),
                                               pow((1 / theta(1)), 2));
    }
  }
  
  return l_likelihood;
}

inline a_double log_likelihood_eigen_IGP_theta(a_vector theta,
                                               arma::vec basis_coef,
                                               const arma::field<arma::mat>& basis_funct,
                                               const arma::field<arma::vec>& X,
                                               const arma::vec& n){
  
  a_double l_likelihood = 0;
  // Calculate log-likelihood for trials
  for(int i = 0; i < n.n_elem; i++){
    for(int j = 0; j < n(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X(i,0)(j), (1 / (theta(0) * std::exp(arma::dot(basis_funct(i,0).row(j), basis_coef)))),
                                               pow((1 / theta(1)), 2));
    }
  }
  
  return l_likelihood;
}

inline a_double log_likelihood_eigen_IGP_basis(arma::vec theta,
                                               a_vector basis_coef,
                                               const arma::field<arma::mat>& basis_funct,
                                               const arma::field<arma::vec>& X,
                                               const arma::vec& n){
  
  a_double l_likelihood = 0;
  // Calculate log-likelihood
  for(int i = 0; i < n.n_elem; i++){
    for(int j = 0; j < n(i); j++){
      l_likelihood = l_likelihood + dinv_gauss(X(i,0)(j), (1 / (theta(0) * CppAD::exp(dot_AD1(basis_funct(i,0).row(j), basis_coef)))),
                                               pow((1 / theta(1)), 2));
    }
  }
  
  return l_likelihood;
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
                               const double& I_A_sigma_sq,
                               const double& I_B_sigma_sq){
  double l_posterior = log_likelihood_TI(Labels, theta, basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                           log_prior_FR(I_A_sigma_sq, I_B_sigma_sq,
                                                        theta, basis_coef_A, basis_coef_B);
  return l_posterior;
}

inline a_double log_posterior_eigen_theta(arma::field<arma::vec>& Labels,
                                        a_vector theta,
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
                                        const double& I_A_mean, 
                                        const double& I_A_shape,
                                        const double& I_B_mean,
                                        const double& I_B_shape,
                                        const double& sigma_A_mean,
                                        const double& sigma_A_shape,
                                        const double& sigma_B_mean,
                                        const double& sigma_B_shape){
  a_double l_posterior = log_likelihood_eigen_theta(Labels, theta, basis_coef_A, basis_coef_B,
                                                  basis_funct_A, basis_funct_B, basis_funct_AB,
                                                  X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                                    log_prior(I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                                              sigma_A_mean, sigma_A_shape,
                                                              sigma_B_mean, sigma_B_shape,
                                                              basis_coef_A, basis_coef_B, theta);
  return l_posterior;
}


inline a_double log_posterior_eigen_basis(arma::field<arma::vec>& Labels,
                                          arma::vec theta,
                                          a_vector basis_coef,
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
                                          const double& I_B_sigma_sq){
  a_double l_posterior = log_likelihood_eigen_basis(Labels, theta, basis_coef,
                                                    basis_funct_A, basis_funct_B, basis_funct_AB,
                                                    X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                                      log_prior_FR(I_A_sigma_sq, I_B_sigma_sq,
                                                                   theta, basis_coef);
  return l_posterior;
}


inline double log_posterior_theta(arma::field<arma::vec>& Labels,
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
                                  const double& I_A_mean, 
                                  const double& I_A_shape,
                                  const double& I_B_mean,
                                  const double& I_B_shape,
                                  const double& sigma_A_mean,
                                  const double& sigma_A_shape,
                                  const double& sigma_B_mean,
                                  const double& sigma_B_shape){
  double l_posterior = log_likelihood_TI(Labels, theta, basis_coef_A, basis_coef_B,
                                         basis_funct_A, basis_funct_B, basis_funct_AB,
                                         X_A, X_B, X_AB, n_A, n_B, n_AB) +
                                           log_prior(I_A_mean, I_A_shape, I_B_mean, I_B_shape, 
                                                     sigma_A_mean, sigma_A_shape,
                                                     sigma_B_mean, sigma_B_shape,
                                                     basis_coef_A, basis_coef_B, theta);
  return l_posterior;
}


inline double log_posterior_IGP_theta(arma::vec theta,
                                      arma::vec basis_coef,
                                      const arma::field<arma::mat>& basis_funct,
                                      const arma::field<arma::vec>& X,
                                      const arma::vec& n,
                                      const double& I_mean,
                                      const double& I_shape,
                                      const double& sigma_mean,
                                      const double& sigma_shape){

  double l_posterior = log_likelihood_IGP_theta(theta, basis_coef, basis_funct, X, n) +
    (dinv_gauss(theta(0), I_mean, I_shape)) +  dinv_gauss(theta(1), sigma_mean, sigma_shape);
  
  return l_posterior;
}

inline a_double log_posterior_eigen_IGP_theta(a_vector theta,
                                              arma::vec basis_coef,
                                              const arma::field<arma::mat>& basis_funct,
                                              const arma::field<arma::vec>& X,
                                              const arma::vec& n,
                                              const double& I_mean,
                                              const double& I_shape,
                                              const double& sigma_mean,
                                              const double& sigma_shape){

  a_double l_posterior = log_likelihood_eigen_IGP_theta(theta, basis_coef, basis_funct, X, n) +
    (dinv_gauss(theta(0), I_mean, I_shape)) +  dinv_gauss(theta(1), sigma_mean, sigma_shape);
  
  return l_posterior;
}

inline double log_posterior_IGP_basis(arma::vec theta,
                                      arma::vec basis_coef,
                                      const arma::field<arma::mat>& basis_funct,
                                      const arma::field<arma::vec>& X,
                                      const arma::vec& n,
                                      const double& I_sigma_sq){
  double l_posterior = log_likelihood_IGP_theta(theta, basis_coef, basis_funct, X, n) +
    log_prior_FR1(I_sigma_sq, theta, basis_coef);
  return l_posterior;
}

inline a_double log_posterior_eigen_IGP_basis(arma::vec theta,
                                              a_vector basis_coef,
                                              const arma::field<arma::mat>& basis_funct,
                                              const arma::field<arma::vec>& X,
                                              const arma::vec& n,
                                              const double& I_sigma_sq){
  a_double l_posterior = log_likelihood_eigen_IGP_basis(theta, basis_coef, basis_funct, X, n) +
    log_prior_FR1(I_sigma_sq, theta, basis_coef);
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
                                           const double& I_A_sigma_sq,
                                           const double& I_B_sigma_sq){
  double l_posterior = log_posterior_FR(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                        basis_funct_A, basis_funct_B, basis_funct_AB,
                                        X_A, X_B, X_AB, n_A, n_B, n_AB, 
                                        I_A_sigma_sq, I_B_sigma_sq);
  return l_posterior;
}



inline double transformed_log_posterior_theta(arma::field<arma::vec>& Labels,
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
                                              const double& I_A_mean, 
                                              const double& I_A_shape,
                                              const double& I_B_mean,
                                              const double& I_B_shape,
                                              const double& sigma_A_mean,
                                              const double& sigma_A_shape,
                                              const double& sigma_B_mean,
                                              const double& sigma_B_shape){
  double l_posterior = log_posterior_theta(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                           basis_funct_A, basis_funct_B, basis_funct_AB,
                                           X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_mean,
                                           I_A_shape, I_B_mean, I_B_shape, sigma_A_mean, 
                                           sigma_A_shape,sigma_B_mean, sigma_B_shape) +
                                             arma::accu(theta.subvec(0,3));
  return l_posterior;
}


inline a_double transformed_log_posterior_eigen_theta(arma::field<arma::vec>& Labels,
                                                      a_vector theta,
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
                                                      const double& I_A_mean, 
                                                      const double& I_A_shape,
                                                      const double& I_B_mean,
                                                      const double& I_B_shape,
                                                      const double& sigma_A_mean,
                                                      const double& sigma_A_shape,
                                                      const double& sigma_B_mean,
                                                      const double& sigma_B_shape){
  for(int i = 0; i < theta.rows(); i++){
    theta(i) = CppAD::exp(theta(i));
  }
  
  a_double l_posterior = log_posterior_eigen_theta(Labels, theta, basis_coef_A, basis_coef_B,
                                                   basis_funct_A, basis_funct_B, basis_funct_AB,
                                                   X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_mean,
                                                   I_A_shape, I_B_mean, I_B_shape, sigma_A_mean, 
                                                   sigma_A_shape,sigma_B_mean, sigma_B_shape) +
                                                     CppAD::log(theta(0)) + CppAD::log(theta(1)) + 
                                                     CppAD::log(theta(2)) + CppAD::log(theta(3));
  return l_posterior;
}


inline a_double transformed_log_posterior_eigen_basis(arma::field<arma::vec>& Labels,
                                                      arma::vec theta,
                                                      a_vector basis_coef,
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
                                                      const double& I_B_sigma_sq){
  a_double l_posterior = log_posterior_eigen_basis(Labels, transform_pars(theta), basis_coef,
                                                   basis_funct_A, basis_funct_B, basis_funct_AB,
                                                   X_A, X_B, X_AB, n_A, n_B, n_AB, 
                                                   I_A_sigma_sq, I_B_sigma_sq);
  return l_posterior;
}

inline double transformed_log_posterior_IGP_theta(arma::vec theta,
                                                  arma::vec& basis_coef,
                                                  const arma::field<arma::mat>& basis_funct,
                                                  const arma::field<arma::vec>& X,
                                                  const arma::vec& n,
                                                  const double& I_mean,
                                                  const double& I_shape,
                                                  const double& sigma_mean,
                                                  const double& sigma_shape){
  double l_posterior = log_posterior_IGP_theta(transform_pars(theta), basis_coef,
                                               basis_funct, X, n, I_mean,
                                               I_shape, sigma_mean, sigma_shape) +
                                                 arma::accu(theta);
  return l_posterior;
}

inline a_double transformed_log_posterior_eigen_IGP_theta(a_vector theta,
                                                          arma::vec& basis_coef,
                                                          const arma::field<arma::mat>& basis_funct,
                                                          const arma::field<arma::vec>& X,
                                                          const arma::vec& n,
                                                          const double& I_mean,
                                                          const double& I_shape,
                                                          const double& sigma_mean,
                                                          const double& sigma_shape){
  for(int i = 0; i < theta.rows(); i++){
    theta(i) = CppAD::exp(theta(i));
  }
  
  a_double l_posterior = log_posterior_eigen_IGP_theta(theta, basis_coef,
                                                       basis_funct, X, n, I_mean,
                                                       I_shape, sigma_mean, sigma_shape) +
                                                         CppAD::log(theta(0)) + CppAD::log(theta(1));
  return l_posterior;
}

inline double transformed_log_posterior_IGP_basis(arma::vec theta,
                                                  arma::vec basis_coef,
                                                  const arma::field<arma::mat>& basis_funct,
                                                  const arma::field<arma::vec>& X,
                                                  const arma::vec& n,
                                                  const double& I_sigma_sq){
  double l_posterior = log_posterior_IGP_basis(transform_pars(theta), basis_coef,
                                               basis_funct, X, n, I_sigma_sq);
  return l_posterior;
}

inline a_double transformed_log_posterior_eigen_IGP_basis(arma::vec theta,
                                                          a_vector basis_coef,
                                                          const arma::field<arma::mat>& basis_funct,
                                                          const arma::field<arma::vec>& X,
                                                          const arma::vec& n,
                                                          const double& I_sigma_sq){

  a_double l_posterior = log_posterior_eigen_IGP_basis(transform_pars(theta), basis_coef,
                                                       basis_funct, X, n, I_sigma_sq);
  return l_posterior;
}


inline arma::vec calc_gradient_eigen_theta(arma::field<arma::vec>& Labels,
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
                                           const double& I_A_mean, 
                                           const double& I_A_shape,
                                           const double& I_B_mean,
                                           const double& I_B_shape,
                                           const double& sigma_A_mean,
                                           const double& sigma_A_shape,
                                           const double& sigma_B_mean,
                                           const double& sigma_B_shape,
                                           ADFun<double>& gr){
  arma::vec grad((theta.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(theta.n_elem);
  a_vector ax(theta.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < theta.n_elem; i++){
    x(i) = theta(i);
    ax(i) = x(i);
    
  }
    
  
  Independent(ax);
  y(0) = log_posterior_eigen_theta(Labels, ax, basis_coef_A, basis_coef_B,
    basis_funct_A, basis_funct_B, basis_funct_AB,
    X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_mean,
    I_A_shape, I_B_mean, I_B_shape, sigma_A_mean, 
    sigma_A_shape,sigma_B_mean, sigma_B_shape);
  gr.Dependent(ax, y);
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad.subvec(0, grad.n_elem -2);
}

inline arma::vec calc_gradient_eigen_theta_update(arma::field<arma::vec>& Labels,
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
                                                  const double& I_A_mean, 
                                                  const double& I_A_shape,
                                                  const double& I_B_mean,
                                                  const double& I_B_shape,
                                                  const double& sigma_A_mean,
                                                  const double& sigma_A_shape,
                                                  const double& sigma_B_mean,
                                                  const double& sigma_B_shape,
                                                  ADFun<double>& gr){
  arma::vec grad((theta.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(theta.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < theta.n_elem; i++){
    x(i) = theta(i);
    
  }
  
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad.subvec(0, grad.n_elem -2);
}

inline arma::vec calc_gradient_eigen_basis(arma::field<arma::vec>& Labels,
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
                                           const double& I_A_sigma_sq,
                                           const double& I_B_sigma_sq,
                                           ADFun<double>& gr){
  arma::vec grad((basis_coef_A.n_elem + basis_coef_B.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(basis_coef_A.n_elem + basis_coef_B.n_elem);
  a_vector ax(basis_coef_A.n_elem + basis_coef_B.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < basis_coef_A.n_elem + basis_coef_B.n_elem; i++){
    if(i < basis_coef_A.n_elem){
      x(i) = basis_coef_A(i);
      ax(i) = x(i);
    }else{
      x(i) = basis_coef_B(i - basis_coef_A.n_elem);
      ax(i) = x(i);
    }
  }
  
  
  Independent(ax);
  y(0) = log_posterior_eigen_basis(Labels, theta, ax,
    basis_funct_A, basis_funct_B, basis_funct_AB,
    X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_sigma_sq, I_B_sigma_sq);
  gr.Dependent(ax, y);
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}

inline arma::vec calc_gradient_eigen_basis_update(arma::field<arma::vec>& Labels,
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
                                                  const double& I_A_sigma_sq,
                                                  const double& I_B_sigma_sq,
                                                  ADFun<double>& gr){
  arma::vec grad((basis_coef_A.n_elem + basis_coef_B.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(basis_coef_A.n_elem + basis_coef_B.n_elem);
  
  for (int i = 0; i < basis_coef_A.n_elem + basis_coef_B.n_elem; i++){
    if(i < basis_coef_A.n_elem){
      x(i) = basis_coef_A(i);
    }else{
      x(i) = basis_coef_B(i - basis_coef_A.n_elem);
    }
  }
  
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}

inline arma::vec calc_gradient_eigen_IGP_theta(arma::vec theta,
                                               arma::vec basis_coef,
                                               const arma::field<arma::mat>& basis_funct,
                                               const arma::field<arma::vec>& X,
                                               const arma::vec& n,
                                               const double& I_mean, 
                                               const double& I_shape,
                                               const double& sigma_mean,
                                               const double& sigma_shape,
                                               ADFun<double>& gr){
  arma::vec grad((theta.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(theta.n_elem);
  a_vector ax(theta.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < theta.n_elem; i++){
    x(i) = theta(i);
    ax(i) = x(i);
    
  }
  
  
  Independent(ax);
  y(0) = log_posterior_eigen_IGP_theta(ax, basis_coef, basis_funct,
    X, n, I_mean, I_shape, sigma_mean, sigma_shape);
  gr.Dependent(ax, y);
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}

inline arma::vec calc_gradient_eigen_IGP_theta_update(arma::vec theta,
                                                      arma::vec basis_coef,
                                                      const arma::field<arma::mat>& basis_funct,
                                                      const arma::field<arma::vec>& X,
                                                      const arma::vec& n,
                                                      const double& I_mean, 
                                                      const double& I_shape,
                                                      const double& sigma_mean,
                                                      const double& sigma_shape,
                                                      ADFun<double>& gr){
  arma::vec grad((theta.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(theta.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < theta.n_elem; i++){
    x(i) = theta(i);
    
  }
  
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}

inline arma::vec calc_gradient_eigen_IGP_basis(arma::vec theta,
                                               arma::vec basis_coef,
                                               const arma::field<arma::mat>& basis_funct,
                                               const arma::field<arma::vec>& X,
                                               const arma::vec& n,
                                               const double& I_sigma_sq,
                                               ADFun<double>& gr){
  arma::vec grad((basis_coef.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(basis_coef.n_elem);
  a_vector ax(basis_coef.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < basis_coef.n_elem; i++){
    x(i) = basis_coef(i);
    ax(i) = x(i);
    
  }
  
  
  Independent(ax);
  y(0) = log_posterior_eigen_IGP_basis(theta, ax, basis_funct,
    X, n, I_sigma_sq);
  gr.Dependent(ax, y);
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}

inline arma::vec calc_gradient_eigen_IGP_basis_update(arma::vec theta,
                                                      arma::vec basis_coef,
                                                      const arma::field<arma::mat>& basis_funct,
                                                      const arma::field<arma::vec>& X,
                                                      const arma::vec& n,
                                                      const double& I_sigma_sq,
                                                      ADFun<double>& gr){
  arma::vec grad((basis_coef.n_elem), arma::fill::zeros);
  Eigen::VectorXd x(basis_coef.n_elem);
  a_vector y(1);
  
  for (int i = 0; i < basis_coef.n_elem; i++){
    x(i) = basis_coef(i);
    
  }
  
  Eigen::VectorXd res = gr.Jacobian(x);
  for(int i = 0; i < res.rows(); i++){
    grad(i) = res(i);
  }
  return grad;
}


inline arma::vec trans_calc_gradient_eigen_theta(arma::field<arma::vec>& Labels,
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
                                                 const double& I_A_mean, 
                                                 const double& I_A_shape,
                                                 const double& I_B_mean,
                                                 const double& I_B_shape,
                                                 const double& sigma_A_mean,
                                                 const double& sigma_A_shape,
                                                 const double& sigma_B_mean,
                                                 const double& sigma_B_shape,
                                                 ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_theta(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                             basis_funct_A, basis_funct_B, basis_funct_AB,
                                             X_A, X_B, X_AB, n_A, n_B, n_AB,
                                             I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                             sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, gr);
  grad = grad + arma::ones(grad.n_elem);
  return grad;
}
inline arma::vec trans_calc_gradient_eigen_theta_update(arma::field<arma::vec>& Labels,
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
                                                        const double& I_A_mean, 
                                                        const double& I_A_shape,
                                                        const double& I_B_mean,
                                                        const double& I_B_shape,
                                                        const double& sigma_A_mean,
                                                        const double& sigma_A_shape,
                                                        const double& sigma_B_mean,
                                                        const double& sigma_B_shape,
                                                        ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_theta_update(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                                    basis_funct_A, basis_funct_B, basis_funct_AB,
                                                    X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                    I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                                    sigma_A_mean, sigma_A_shape,sigma_B_mean, sigma_B_shape, gr);
  grad = grad + arma::ones(grad.n_elem);
  return grad;
}

inline arma::vec trans_calc_gradient_eigen_basis(arma::field<arma::vec>& Labels,
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
                                                        const double& I_A_sigma_sq,
                                                        const double& I_B_sigma_sq,
                                                        ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_basis(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                             basis_funct_A, basis_funct_B, basis_funct_AB,
                                             X_A, X_B, X_AB, n_A, n_B, n_AB,
                                             I_A_sigma_sq, I_B_sigma_sq, gr);
  return grad;
}


inline arma::vec trans_calc_gradient_eigen_basis_update(arma::field<arma::vec>& Labels,
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
                                                        const double& I_A_sigma_sq,
                                                        const double& I_B_sigma_sq,
                                                        ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_basis_update(Labels, transform_pars(theta), basis_coef_A, basis_coef_B,
                                                    basis_funct_A, basis_funct_B, basis_funct_AB,
                                                    X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                    I_A_sigma_sq, I_B_sigma_sq, gr);
  return grad;
}

inline arma::vec trans_calc_gradient_eigen_IGP_theta(arma::vec theta,
                                                     arma::vec basis_coef,
                                                     const arma::field<arma::mat>& basis_funct,
                                                     const arma::field<arma::vec>& X,
                                                     const arma::vec& n,
                                                     const double& I_mean, 
                                                     const double& I_shape,
                                                     const double& sigma_mean,
                                                     const double& sigma_shape,
                                                     ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_IGP_theta(transform_pars(theta), basis_coef, basis_funct,
                                                 X, n, I_mean, I_shape, sigma_mean, sigma_shape, gr);
  grad = grad + arma::ones(grad.n_elem);
  return grad;
}

inline arma::vec trans_calc_gradient_eigen_IGP_theta_update(arma::vec theta,
                                                            arma::vec basis_coef,
                                                            const arma::field<arma::mat>& basis_funct,
                                                            const arma::field<arma::vec>& X,
                                                            const arma::vec& n,
                                                            const double& I_mean, 
                                                            const double& I_shape,
                                                            const double& sigma_mean,
                                                            const double& sigma_shape,
                                                            ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_IGP_theta_update(transform_pars(theta), basis_coef, basis_funct,
                                                        X, n, I_mean, I_shape, sigma_mean, sigma_shape, gr);
  grad = grad + arma::ones(grad.n_elem);
  return grad;
}

inline arma::vec trans_calc_gradient_eigen_IGP_basis(arma::vec theta,
                                                     arma::vec basis_coef,
                                                     const arma::field<arma::mat>& basis_funct,
                                                     const arma::field<arma::vec>& X,
                                                     const arma::vec& n,
                                                     const double& I_sigma_sq,
                                                     ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_IGP_basis(transform_pars(theta), basis_coef, basis_funct,
                                                 X, n, I_sigma_sq, gr);
  return grad;
}

inline arma::vec trans_calc_gradient_eigen_IGP_basis_update(arma::vec theta,
                                                            arma::vec basis_coef,
                                                            const arma::field<arma::mat>& basis_funct,
                                                            const arma::field<arma::vec>& X,
                                                            const arma::vec& n,
                                                            const double& I_sigma_sq,
                                                            ADFun<double>& gr){
  arma::vec grad = calc_gradient_eigen_IGP_basis_update(transform_pars(theta), basis_coef, basis_funct,
                                                        X, n, I_sigma_sq, gr);
  return grad;
}

inline double log_posterior_model(double log_lik,
                                  arma::vec theta,
                                  const double& I_A_mean, 
                                  const double& I_A_shape,
                                  const double& I_B_mean,
                                  const double& I_B_shape,
                                  const double& sigma_A_mean,
                                  const double& sigma_A_shape,
                                  const double& sigma_B_mean,
                                  const double& sigma_B_shape,
                                  const double delta_shape,
                                  const double delta_rate){
  double lposterior = log_lik;
  arma::vec basis_coef_ph_A = arma::zeros(1);
  arma::vec basis_coef_ph_B = arma::zeros(1);
  lposterior = lposterior + log_prior(I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                      sigma_A_mean, sigma_A_shape, sigma_B_mean,
                                      sigma_B_shape, basis_coef_ph_A, basis_coef_ph_B,
                                      theta);
  lposterior = lposterior + R::dgamma(theta(4), delta_shape, (1 / delta_rate), true);
  
  return lposterior;
}


inline double log_posterior_model_TI(double log_lik,
                                     arma::vec theta,
                                     arma::vec basis_coef_A,
                                     arma::vec basis_coef_B,
                                     double I_A_sigma_sq,
                                     double I_B_sigma_sq,
                                     const double& I_A_mean, 
                                     const double& I_A_shape,
                                     const double& I_B_mean,
                                     const double& I_B_shape,
                                     const double& sigma_A_mean,
                                     const double& sigma_A_shape,
                                     const double& sigma_B_mean,
                                     const double& sigma_B_shape,
                                     const double delta_shape,
                                     const double delta_rate,
                                     const double alpha,
                                     const double beta){
  double lposterior = log_lik;
  lposterior = lposterior + log_prior(I_A_mean, I_A_shape, I_B_mean, I_B_shape,
                                      sigma_A_mean, sigma_A_shape, sigma_B_mean,
                                      sigma_B_shape, basis_coef_A, basis_coef_B, theta);
  lposterior = lposterior + R::dgamma(theta(4), delta_shape, (1 / delta_rate), true);
  
  lposterior = lposterior + log_prior_FR(I_A_sigma_sq, I_B_sigma_sq, theta, basis_coef_A,
                                         basis_coef_B);
  
  lposterior = lposterior + R::dgamma(1/I_A_sigma_sq, alpha, 1/ beta, true);
  lposterior = lposterior + R::dgamma(1/I_B_sigma_sq, alpha, 1/ beta, true);
  
  return lposterior;
}

inline double log_posterior_IGP_model(double log_lik,
                                      arma::vec theta,
                                      const double& I_mean, 
                                      const double& I_shape,
                                      const double& sigma_mean,
                                      const double& sigma_shape){
  double lposterior = log_lik;
  lposterior = lposterior + dinv_gauss(theta(0), I_mean, I_shape) + 
    dinv_gauss(theta(1), sigma_mean, sigma_shape);
  return lposterior;
}

inline double log_posterior_IGP_model_TI(double log_lik,
                                         arma::vec theta,
                                         arma::vec basis_coef,
                                         double I_sigma_sq,
                                         const double& I_mean, 
                                         const double& I_shape,
                                         const double& sigma_mean,
                                         const double& sigma_shape,
                                         const double alpha,
                                         const double beta){
  double lposterior = log_lik;
  
  // prior on I and sigma
  lposterior = lposterior + (dinv_gauss(theta(0), I_mean, I_shape)) + 
    dinv_gauss(theta(1), sigma_mean, sigma_shape);
  // prior on b-splines
  lposterior = lposterior + log_prior_FR1(I_sigma_sq, theta, basis_coef);
  // prior on I_sigma_sq
  lposterior = lposterior + R::dgamma(1/I_sigma_sq, alpha, 1/ beta, true);
  
  
  return lposterior;
}

}


#endif