#ifndef WAIC_H
#define WAIC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"

namespace NeuralComp {

inline arma::mat approximate_L_given_theta(arma::vec theta,
                                           double max_time,
                                           int n_eval){
  arma::mat P_mat(2, 2, arma::fill::zeros);
  arma::vec eval = arma::linspace(0, max_time, n_eval);
  double step_size = eval(1) - eval(0);
  
  for(int i = 0; i < n_eval; i++){
    P_mat(0,0) = P_mat(0,0) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
      dinv_gauss(eval(i), (1 / theta(0)), pow((1 / theta(2)), 2))) * step_size;
    P_mat(1,1) = P_mat(1,1) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
      dinv_gauss(eval(i), (1 / theta(1)), pow((1 / theta(3)), 2))) * step_size;
  }
  
  eval = arma::linspace(theta(4), max_time + theta(4), n_eval);
  for(int i = 0; i < n_eval; i++){
    P_mat(0,1) = P_mat(0,1) + std::exp(pinv_gauss(eval(i), (1 / theta(0)), pow((1 / theta(2)), 2)) +
      dinv_gauss(eval(i) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2))) * step_size;
    P_mat(1,0) = P_mat(1,0) + std::exp(pinv_gauss(eval(i), (1 / theta(1)), pow((1 / theta(3)), 2)) +
      dinv_gauss(eval(i) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2))) * step_size;
  }
  
  P_mat(0,0) = (P_mat(0,0) + (1 - P_mat(0,1))) / 2;
  P_mat(0,1) = 1 - P_mat(0,0);
  P_mat(1,0) = (P_mat(1,0) + (1 - P_mat(1,1))) / 2;
  P_mat(1,1) = 1 - P_mat(1,0);
  
  return P_mat;
  
}

inline arma::vec calc_loglikelihood_A(const arma::vec X_A,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_A,
                                      const arma::mat basis_funct_A,
                                      const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec llik = arma::zeros(n_MCMC - burnin_num);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X_A.n_elem; j++){
      llik(i - burnin_num) = llik(i - burnin_num) + dinv_gauss(X_A(j), (1 / (theta(i, 0) + arma::dot(basis_funct_A.row(j), basis_coef_A.row(i)))),
                                               pow((1 / theta(i, 2)), 2));
    }
  }
    
  return llik;
}

inline arma::vec calc_loglikelihood_B(const arma::vec X_B,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_B,
                                      const arma::mat basis_funct_B,
                                      const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec llik = arma::zeros(n_MCMC - burnin_num);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X_B.n_elem; j++){
      llik(i - burnin_num) = llik(i - burnin_num) + dinv_gauss(X_B(j), (1 / (theta(i, 1) + arma::dot(basis_funct_B.row(j), basis_coef_B.row(i)))),
           pow((1 / theta(i, 3)), 2));
    }
  }
  
  return llik;
}

inline arma::vec calc_loglikelihood_AB(const arma::vec X_AB,
                                       const arma::mat theta,
                                       const arma::mat basis_coef_A,
                                       const arma::mat basis_coef_B,
                                       const arma::mat basis_funct_AB,
                                       const arma::field<arma::mat> Labels,
                                       arma::field<arma::mat> P_mat,
                                       arma::field<arma::mat> P_mat_0,
                                       const int obs_num,
                                       const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec llik = arma::zeros(n_MCMC - burnin_num);
  
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X_AB.n_elem; j++){
      if(Labels(obs_num, 0)(i,j) == 0){
        // label is A
        if(j != 0){
          if(Labels(obs_num, 0)(i,j-1) == 0){
            // Condition if spike has not switched (still in A)
            llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j) - theta(i, 4), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))),
                 pow((1 / theta(i, 3)), 2)) +
                   dinv_gauss(X_AB(j), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat(i - burnin_num, 0)(0, 0)));
          }else{
            // Condition if spike has switched from B to A
            llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))), pow((1 / theta(i, 3)), 2)) +
              dinv_gauss(X_AB(j) - theta(i, 4), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat(i - burnin_num, 0)(1, 0)));
          }
        }else{
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))), pow((1 / theta(i, 3)), 2)) +
            dinv_gauss(X_AB(j), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat_0(i - burnin_num, 0)(0, 0)));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(obs_num, 0)(i, j-1) == 1){
            // Condition if spike has not switched (still in B)
            llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j) - theta(i, 4), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2)) +
              dinv_gauss(X_AB(j), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat(i - burnin_num, 0)(1, 1)));
          }else{
            // Condition if spike has switched from A to B
            llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2)) +
              dinv_gauss(X_AB(j) - theta(i, 4), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat(i - burnin_num, 0)(0, 1)));
          }
        }else{
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(j), (1 / (theta(i, 0) + arma::dot(basis_funct_AB.row(j), basis_coef_A.row(i)))), pow((1 / theta(i, 2)), 2)) +
            dinv_gauss(X_AB(j), (1 / (theta(i, 1) + arma::dot(basis_funct_AB.row(j), basis_coef_B.row(i)))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat_0(i - burnin_num, 0)(1, 1)));
        }
      }
    }
  }
  return llik;
}

inline double calc_WAIC_competition(const arma::field<arma::vec> X_A,
                                    const arma::field<arma::vec> X_B,
                                    const arma::field<arma::vec> X_AB,
                                    const arma::vec n_A,
                                    const arma::vec n_B,
                                    const arma::vec n_AB,
                                    const arma::mat theta,
                                    const arma::mat basis_coef_A,
                                    const arma::mat basis_coef_B,
                                    const arma::field<arma::mat> basis_funct_A,
                                    const arma::field<arma::mat> basis_funct_B,
                                    const arma::field<arma::mat> basis_funct_AB,
                                    const arma::field<arma::mat> Labels,
                                    const double burnin_prop,
                                    const double max_time,
                                    const int n_eval){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  
  // Placeholder for log-likelihood by observation
  arma::mat llik_A = arma::zeros(n_A.n_elem, n_MCMC - burnin_num);
  arma::mat llik_B = arma::zeros(n_B.n_elem, n_MCMC - burnin_num);
  arma::mat llik_AB = arma::zeros(n_AB.n_elem, n_MCMC - burnin_num);
  
  arma::field<arma::mat> P_mat(n_MCMC - burnin_num, 1);
  arma::field<arma::mat> P_mat0(n_MCMC - burnin_num, 1);
  arma::mat theta_0 = theta;
  theta_0.col(4) = arma::zeros(theta_0.n_rows);
  for(int i = burnin_num; i < n_MCMC; i++){
    P_mat(i - burnin_num, 0) = approximate_L_given_theta(theta.row(i).t(), max_time, n_eval);
    P_mat0(i - burnin_num, 0) = approximate_L_given_theta(theta_0.row(i).t(), max_time, n_eval);
  }
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A.row(i) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop).t();
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B.row(i) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop).t();
  }
  
  Rcpp::Rcout << "Made it";
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB.row(i) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
               basis_funct_AB(i,0), Labels, P_mat, P_mat0, i, burnin_prop).t();
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_A.row(i))));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_B.row(i))));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_AB.row(i))));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A.row(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B.row(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB.row(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}


inline arma::vec calc_loglikelihood_IGP(const arma::vec X,
                                        const arma::mat theta,
                                        const arma::mat basis_coef,
                                        const arma::mat basis_funct,
                                        const double burnin_prop){
  
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec llik = arma::zeros(n_MCMC - burnin_num);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X.n_elem; j++){
      llik(i - burnin_num) = llik(i - burnin_num) + dinv_gauss(X(j), (1 / (theta(i, 0) + arma::dot(basis_funct.row(j), basis_coef.row(i)))),
           pow((1 / theta(i, 1)), 2));
    }
  }
  
  return llik;
}

inline double calc_WAIC_IGP(const arma::field<arma::vec> X_A,
                            const arma::field<arma::vec> X_B,
                            const arma::field<arma::vec> X_AB,
                            const arma::vec n_A,
                            const arma::vec n_B,
                            const arma::vec n_AB,
                            const arma::mat theta_A,
                            const arma::mat basis_coef_A,
                            const arma::mat theta_B,
                            const arma::mat basis_coef_B,
                            const arma::mat theta_AB,
                            const arma::mat basis_coef_AB,
                            const arma::field<arma::mat> basis_funct_A,
                            const arma::field<arma::mat> basis_funct_B,
                            const arma::field<arma::mat> basis_funct_AB,
                            const double burnin_prop){
  int n_MCMC_A = theta_A.n_rows;
  int burnin_num_A = n_MCMC_A - std::floor((1 - burnin_prop) * n_MCMC_A);
  int n_MCMC_B = theta_B.n_rows;
  int burnin_num_B = n_MCMC_B - std::floor((1 - burnin_prop) * n_MCMC_B);
  int n_MCMC_AB = theta_AB.n_rows;
  int burnin_num_AB = n_MCMC_AB - std::floor((1 - burnin_prop) * n_MCMC_AB);
  
  // Placeholder for log-likelihood by observation
  arma::mat llik_A = arma::zeros(n_A.n_elem, n_MCMC_A - burnin_num_A);
  arma::mat llik_B = arma::zeros(n_B.n_elem, n_MCMC_B - burnin_num_B);
  arma::mat llik_AB = arma::zeros(n_AB.n_elem, n_MCMC_AB - burnin_num_AB);
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A.row(i) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop).t();
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B.row(i) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop).t();
  }
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB.row(i) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
                basis_funct_AB(i,0), burnin_prop).t();
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_A.row(i))));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_B.row(i))));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_AB.row(i))));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A.row(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B.row(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB.row(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}


}


#endif