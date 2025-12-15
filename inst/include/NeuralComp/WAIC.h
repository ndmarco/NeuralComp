#ifndef WAIC_H
#define WAIC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include "Labels.h"
#include <splines2Armadillo.h>

namespace NeuralComp {

inline arma::mat calc_loglikelihood_A(const arma::vec X_A,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_A,
                                      const arma::mat basis_funct_A,
                                      const double burnin_prop,
                                      const double& end_time){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik;
  if(X_A.n_elem > 0){
    llik = arma::zeros(n_MCMC - burnin_num, X_A.n_elem);
    for(int i = burnin_num; i < n_MCMC; i++){
      for(arma::uword j = 0; j < X_A.n_elem; j++){
        llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X_A(j), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_A.row(j), basis_coef_A.row(i))))),
             std::pow((1 / theta(i, 2)), 2));
      }
      llik(i - burnin_num, X_A.n_elem - 1) = llik(i - burnin_num, X_A.n_elem - 1) + 
        pinv_gauss(end_time - arma::accu(X_A), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_A.row(X_A.n_elem), basis_coef_A.row(i))))),
                   std::pow((1 / theta(i,2)), 2.0));
    }
  }else{
    llik = arma::zeros(n_MCMC - burnin_num, 1);
    for(int i = burnin_num; i < n_MCMC; i++){
      llik(i - burnin_num, 0) = llik(i - burnin_num, 0) + 
        pinv_gauss(end_time, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_A.row(X_A.n_elem), basis_coef_A.row(i))))),
                   std::pow((1 / theta(i,2)), 2.0));
    }
  }
  
  
  return llik;
}

inline arma::mat calc_loglikelihood_B(const arma::vec X_B,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_B,
                                      const arma::mat basis_funct_B,
                                      const double burnin_prop,
                                      const double& end_time){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik;
  if(X_B.n_elem > 0){
    llik = arma::zeros(n_MCMC - burnin_num, X_B.n_elem);
    for(int i = burnin_num; i < n_MCMC; i++){
      for(arma::uword j = 0; j < X_B.n_elem; j++){
        llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X_B(j), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_B.row(j), basis_coef_B.row(i))))),
             std::pow((1 / theta(i, 3)), 2));
      }
      llik(i - burnin_num, X_B.n_elem - 1) = llik(i - burnin_num, X_B.n_elem - 1) + 
        pinv_gauss(end_time - arma::accu(X_B), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_B.row(X_B.n_elem), basis_coef_B.row(i))))),
                   std::pow((1 / theta(i,3)), 2.0));
    }
  }else{
    llik = arma::zeros(n_MCMC - burnin_num, 1);
    for(int i = burnin_num; i < n_MCMC; i++){
      llik(i - burnin_num, 0) = llik(i - burnin_num, 0) + 
        pinv_gauss(end_time, (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_B.row(X_B.n_elem), basis_coef_B.row(i))))),
                   std::pow((1 / theta(i,3)), 2.0));
    }
  }
  
  return llik;
}

inline arma::mat calc_loglikelihood_IGP(const arma::vec X,
                                        const arma::mat theta,
                                        const arma::mat basis_coef,
                                        const arma::mat basis_funct,
                                        const double burnin_prop,
                                        const double& end_time){
  
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik;
  if(X.n_elem > 0){
    llik = arma::zeros(n_MCMC - burnin_num, X.n_elem);
    for(int i = burnin_num; i < n_MCMC; i++){
      for(arma::uword j = 0; j < X.n_elem; j++){
        llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X(j), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct.row(j), basis_coef.row(i))))),
             std::pow((1 / theta(i, 1)), 2));
      }
      // probability of not observing a spike in the rest of the time
      llik(i - burnin_num, X.n_elem - 1) = llik(i - burnin_num, X.n_elem - 1) + 
        pinv_gauss(end_time - arma::accu(X), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct.row(X.n_elem), basis_coef.row(i))))),
                   std::pow((1 / theta(i,1)), 2.0));
    }
  }else{
    llik = arma::zeros(n_MCMC - burnin_num, 1);
    for(int i = burnin_num; i < n_MCMC; i++){
      // probability of not observing a spike in the rest of the time
      llik(i - burnin_num, 0) = llik(i - burnin_num, 0) + 
        pinv_gauss(end_time, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct.row(X.n_elem), basis_coef.row(i))))),
                   std::pow((1 / theta(i,1)), 2.0));
    }
    
  }
  
  return llik;
}

inline Rcpp::List calc_WAIC_IGP(const arma::field<arma::vec> X_A,
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
                                const double burnin_prop,
                                const double& end_time){
  
  // Placeholder for log-likelihood by observation
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  // calculate log-likelihood for A
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for AB
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
                basis_funct_AB(i,0), burnin_prop, end_time);
  }
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
    }
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
    }
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
    }
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
  double pwaic = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      pwaic = pwaic + arma::var(llik_A(i,0).col(j));
    }
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      pwaic = pwaic + arma::var(llik_B(i,0).col(j));
    }
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
    }
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (lppd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
                                          Rcpp::Named("lppd", lppd),
                                          Rcpp::Named("Effective_pars", pwaic),
                                          Rcpp::Named("llik_A", llik_A),
                                          Rcpp::Named("llik_B", llik_B),
                                          Rcpp::Named("llik_AB", llik_AB));
  return output1;
}

inline double calc_log_mean(arma::vec x){
  double max_val = x.max();
  double inner_sum = 0;
  for(arma::uword i = 0; i < x.n_elem; i++){
    inner_sum = inner_sum + std::exp(x(i) - max_val);
  }
  double output = max_val + std::log(inner_sum) - std::log(x.n_elem);
  return output;
}

inline Rcpp::List calc_WAIC_IGP_observation(const arma::field<arma::vec> X_A,
                                            const arma::field<arma::vec> X_B,
                                            const arma::field<arma::vec> X_AB,
                                            const arma::vec n_A,
                                            const arma::vec n_B,
                                            const arma::vec n_AB,
                                            const double& end_time,
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
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  // calculate log-likelihood for A
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for AB
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
            basis_funct_AB(i,0), burnin_prop, end_time);
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  arma::mat llik_AB_obs = arma::zeros(n_MCMC_AB - burnin_num_AB, n_AB.n_elem);
  
  for(int i = 0; i < n_MCMC_A - burnin_num_A; i++){
    for(arma::uword j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_B - burnin_num_B; i++){
    for(arma::uword j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_AB - burnin_num_AB; i++){
    for(arma::uword j = 0; j < n_AB.n_elem; j++){
      llik_AB_obs(i, j) = arma::accu(llik_AB(j,0).row(i));
    }
  }

  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
  double pwaic = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B_obs.col(i));
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (lppd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  
  Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
                                          Rcpp::Named("lppd", lppd),
                                          Rcpp::Named("Effective_pars", pwaic),
                                          Rcpp::Named("llik_A", llik_A),
                                          Rcpp::Named("llik_B", llik_B),
                                          Rcpp::Named("llik_AB", llik_AB));
  return output1;
}


inline void calc_loglikelihood_AB_Marginal(const arma::vec X_AB,
                                           const arma::mat theta,
                                           const arma::mat basis_coef_A,
                                           const arma::mat basis_coef_B,
                                           const arma::mat basis_funct_AB,
                                           const double burnin_prop,
                                           arma::vec& llik,
                                           const double& end_time){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::field<arma::mat> ph;
  arma::vec theta_i = arma::zeros(theta.n_cols);
  arma::vec basis_coef_A_i = arma::zeros(basis_coef_A.n_cols);
  arma::vec basis_coef_B_i = arma::zeros(basis_coef_B.n_cols);
  for(int i = burnin_num; i < n_MCMC; i++){
    theta_i = theta.row(i).t();
    basis_coef_A_i = basis_coef_A.row(i).t();
    basis_coef_B_i = basis_coef_B.row(i).t();
    if(X_AB.n_elem > 0){
      ph = forward_filtration_delta_int(theta_i, basis_coef_A_i, basis_coef_B_i,
                                        basis_funct_AB, X_AB, end_time);
      llik(i - burnin_num) = arma::accu(ph(1,0));
    }else{
      llik(i - burnin_num) = pinv_gauss(end_time, (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0)) +
        pinv_gauss(end_time, (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0));
    }
   
  }
}



inline Rcpp::List calc_WAIC_competition_Marginal(const arma::field<arma::vec> X_A,
                                                 const arma::field<arma::vec> X_B,
                                                 const arma::field<arma::vec> X_AB,
                                                 const arma::vec n_A,
                                                 const arma::vec n_B,
                                                 const arma::vec n_AB,
                                                 const double& end_time,
                                                 const arma::mat theta,
                                                 const arma::mat basis_coef_A,
                                                 const arma::mat basis_coef_B,
                                                 const arma::field<arma::mat> basis_funct_A,
                                                 const arma::field<arma::mat> basis_funct_B,
                                                 const arma::field<arma::mat> basis_funct_AB,
                                                 const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  
  // Placeholder for log-likelihood by observation
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::vec> llik_AB(n_AB.n_elem, 1);
  
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num);
  }
  
  // calculate log-likelihood for A
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  Rcpp::List output;
  
  // calculate log-likelihood for AB
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    calc_loglikelihood_AB_Marginal(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
                                   basis_funct_AB(i,0), burnin_prop,
                                   llik_AB(i,0), end_time);
    Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC - burnin_num, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC - burnin_num, n_B.n_elem);
  arma::mat llik_AB_obs = arma::zeros(n_MCMC - burnin_num, n_AB.n_elem);
  
  for(int i = 0; i < n_MCMC - burnin_num; i++){
    for(arma::uword j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
    for(arma::uword j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
    for(arma::uword j = 0; j < n_AB.n_elem; j++){
      llik_AB_obs(i, j) = llik_AB(j,0)(i);
    }
  }
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
  double pwaic = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B_obs.col(i));
  }
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (lppd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  
  Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
                                          Rcpp::Named("lppd", lppd),
                                          Rcpp::Named("Effective_pars", pwaic),
                                          Rcpp::Named("llik_A", llik_A_obs),
                                          Rcpp::Named("llik_B", llik_B_obs),
                                          Rcpp::Named("llik_AB", llik_AB_obs));
  return output1;
}


inline void calc_loglikelihood_AB_Marginal_Observation(const arma::vec X_AB,
                                                       const arma::mat theta,
                                                       const arma::mat basis_coef_A,
                                                       const arma::mat basis_coef_B,
                                                       const arma::mat basis_funct_AB,
                                                       const double burnin_prop,
                                                       arma::mat& llik,
                                                       const double& end_time){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::field<arma::mat> ph;
  arma::vec theta_i = arma::zeros(theta.n_cols);
  arma::vec basis_coef_A_i = arma::zeros(basis_coef_A.n_cols);
  arma::vec basis_coef_B_i = arma::zeros(basis_coef_B.n_cols);
  for(int i = burnin_num; i < n_MCMC; i++){
    theta_i = theta.row(i).t();
    basis_coef_A_i = basis_coef_A.row(i).t();
    basis_coef_B_i = basis_coef_B.row(i).t();
    ph = forward_filtration_delta_int(theta_i, basis_coef_A_i, basis_coef_B_i,
                                      basis_funct_AB, X_AB, end_time);
    llik.row(i - burnin_num) = ph(1,0).t();
  }
}


inline double calc_Diff_LPPD_A_B(const arma::field<arma::vec> X_A,
                                 const arma::field<arma::vec> X_B,
                                 const arma::field<arma::vec> X_joint,
                                 const arma::vec n_A,
                                 const arma::vec n_B,
                                 const arma::vec n_joint,
                                 const double& end_time,
                                 const arma::mat theta_A,
                                 const arma::mat basis_coef_A,
                                 const arma::mat theta_B,
                                 const arma::mat basis_coef_B,
                                 const arma::mat theta_joint,
                                 const arma::mat basis_coef_joint,
                                 const arma::field<arma::mat> basis_funct_A,
                                 const arma::field<arma::mat> basis_funct_B,
                                 const arma::field<arma::mat> basis_funct_joint,
                                 const double burnin_prop){
  int n_MCMC_A = theta_A.n_rows;
  int burnin_num_A = n_MCMC_A - std::floor((1 - burnin_prop) * n_MCMC_A);
  int n_MCMC_B = theta_B.n_rows;
  int burnin_num_B = n_MCMC_B - std::floor((1 - burnin_prop) * n_MCMC_B);
  int n_MCMC_joint = theta_joint.n_rows;
  int burnin_num_joint = n_MCMC_joint - std::floor((1 - burnin_prop) * n_MCMC_joint);
  
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_joint(n_joint.n_elem, 1);
  
  // calculate log-likelihood for A
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for joint
  for(arma::uword i = 0; i < n_joint.n_elem; i++){
    llik_joint(i,0) = calc_loglikelihood_IGP(X_joint(i,0), theta_joint, basis_coef_joint, 
            basis_funct_joint(i,0), burnin_prop, end_time);
  }
  
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  
  arma::mat llik_joint_obs = arma::zeros(n_MCMC_joint - burnin_num_joint, n_joint.n_elem);
  
  for(int i = 0; i < n_MCMC_A - burnin_num_A; i++){
    for(arma::uword j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_B - burnin_num_B; i++){
    for(arma::uword j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_joint - burnin_num_joint; i++){
    for(arma::uword j = 0; j < n_joint.n_elem; j++){
      llik_joint_obs(i, j) = arma::accu(llik_joint(j,0).row(i));
    }
  }
  
  // calculate log pointwise predictive density
  double lppd_seperate = 0;
  
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    lppd_seperate = lppd_seperate + calc_log_mean(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    lppd_seperate = lppd_seperate + calc_log_mean(llik_B_obs.col(i));
  }
 
  
  double lppd_joint = 0;
  for(arma::uword i = 0; i < n_joint.n_elem; i++){
    lppd_joint = lppd_joint + calc_log_mean(llik_joint_obs.col(i));
  }
  double ratio_lppd = lppd_seperate - lppd_joint;
  return ratio_lppd;
}

inline Rcpp::List sim_IIGPP(const arma::vec theta,
                            const arma::vec basis_coef,
                            const int n_spike_trains,
                            const double trial_time,
                            const int& basis_degree,
                            const arma::vec& boundary_knots,
                            const arma::vec& internal_knots,
                            const bool& time_inhomogeneous){
  double ISI;
  arma::field<arma::vec> X_sim(n_spike_trains, 1);
  arma::vec n_sim = arma::zeros(n_spike_trains);
  splines2::BSpline bspline;
  arma::vec time = arma::zeros(1);
  arma::field<arma::mat> basis_funct(n_spike_trains,1);
  double total_time;
  int j;
  for(int i = 0; i < n_spike_trains; i++){
    total_time = 0;
    time(0) = total_time;
    X_sim(i,0) = arma::zeros(1);
    j = 0;
    if(time_inhomogeneous == true){
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      arma::mat bspline_mat{bspline.basis(false)};
      ISI = rinv_gauss((1/(theta(0) * std::exp(arma::dot(bspline_mat.row(0), basis_coef)))), (1 / theta(1)) * (1 / theta(1)));
    }else{
      ISI = rinv_gauss((1/(theta(0))), (1 / theta(1)) * (1 / theta(1)));
    }
    X_sim(i,0)(0) = ISI;
    total_time = total_time + ISI;
    j = j + 1;
    while(total_time < trial_time){
      time(0) = total_time;
      if(time_inhomogeneous == true){
        bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                    boundary_knots);
        arma::mat bspline_mat{bspline.basis(false)};
        ISI = rinv_gauss((1/(theta(0) * std::exp(arma::dot(bspline_mat.row(0), basis_coef)))), (1 / theta(1)) * (1 / theta(1)));
      }else{
        ISI = rinv_gauss((1/(theta(0))), (1 / theta(1)) * (1 / theta(1)));
      }
      total_time = total_time + ISI;
      if(total_time < trial_time){
        X_sim(i,0).resize(j + 1);
        X_sim(i,0)(j) = ISI;
        j = j + 1;
      }
    }
    
    
    n_sim(i) = j;
    arma::vec time = arma::zeros(n_sim(i) + 1);
    for(int j = 1; j < (n_sim(i) + 1); j++){
      time(j) = arma::accu(X_sim(i,0).subvec(0,j-1));
    }
    bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                boundary_knots);
    arma::mat bspline_mat{bspline.basis(false)};
    basis_funct(i,0) = bspline_mat;
  }
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("X_sim", X_sim),
                                         Rcpp::Named("n_sim", n_sim),
                                         Rcpp::Named("basis_funct_sim", basis_funct));
  return output;
}

inline double calc_discrepency(const arma::field<arma::vec>& X,
                               const arma::vec& n,
                               const arma::vec theta,
                               const arma::vec basis_coef,
                               const arma::field<arma::mat>& basis_funct,
                               const bool& time_inhomogeneous,
                               const double& end_time){
  double d = 0;
  double mean = 0;
  double lambda;
  for(arma::uword i = 0; i < n.n_elem; i++){
    for(int j = 0; j < n(i); j++){
      if(time_inhomogeneous == true){
        mean = 1 / (theta(0) * std::exp(arma::dot(basis_funct(i,0).row(j), basis_coef)));
        lambda = (1 / theta(1)) * (1 / theta(1));
        d = d + dinv_gauss(X(i,0)(j), mean, lambda);
      }else{
        mean = 1 / theta(0);
        lambda = (1 / theta(1)) * (1 / theta(1));
        d = d + dinv_gauss(X(i,0)(j), mean, lambda);
      }
    }
    if(time_inhomogeneous == true){
      mean = 1 / (theta(0) * std::exp(arma::dot(basis_funct(i,0).row(n(i)), basis_coef)));
      lambda = (1 / theta(1)) * (1 / theta(1));
    }else{
      mean = 1 / theta(0);
      lambda = (1 / theta(1)) * (1 / theta(1));
    }
    // last spike
    d = d + pinv_gauss(end_time - arma::accu(X(i,0)), mean, lambda);
  }
  return d;
}

inline Rcpp::List calc_chi_squared_IIGPP(const arma::field<arma::vec>& X,
                                         const arma::vec& n,
                                         const double& end_time,
                                         const arma::mat& theta,
                                         const arma::mat& basis_coef,
                                         const arma::field<arma::mat>& basis_funct,
                                         const double& trial_time,
                                         const int& basis_degree,
                                         const arma::vec& boundary_knots,
                                         const arma::vec& internal_knots,
                                         const bool& time_inhomogeneous,
                                         const double& burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec D_obs = arma::zeros(n_MCMC - burnin_num);
  arma::vec D_sim = arma::zeros(n_MCMC - burnin_num);
  Rcpp::List sim_output;
  arma::field<arma::vec> X_sim_out;
  arma::vec n_sim_out;
  arma::vec n_var_sim = arma::zeros(n_MCMC - burnin_num);
  arma::vec n_mean_sim = arma::zeros(n_MCMC - burnin_num);
  int n_bigger = 0;
  //Rcpp::Rcout << "Made it 1";
  for(int i = burnin_num; i < n_MCMC; i++){
    sim_output = sim_IIGPP(theta.row(i).t(), basis_coef.row(i).t(), n.n_elem, trial_time,
                           basis_degree, boundary_knots, internal_knots, time_inhomogeneous);
    //Rcpp::Rcout << "Made it 2";
    arma::field<arma::vec> X_sim = sim_output["X_sim"];
    arma::vec n_sim = sim_output["n_sim"];
    arma::field<arma::mat> basis_funct_sim = sim_output["basis_funct_sim"];
    D_obs(i - burnin_num) = calc_discrepency(X, n, theta.row(i).t(), basis_coef.row(i).t(), basis_funct, time_inhomogeneous, end_time);
    D_sim(i - burnin_num) = calc_discrepency(X_sim, n_sim, theta.row(i).t(), basis_coef.row(i).t(), basis_funct_sim, time_inhomogeneous, end_time);
    // if((i % 100) == 0){
    //   Rcpp::Rcout << D_obs(i - burnin_num);
    //   Rcpp::Rcout << "   " << D_sim(i - burnin_num);
    // }
    //Rcpp::Rcout << "Made it 3";
    if(D_obs(i - burnin_num) > D_sim(i - burnin_num)){
      n_bigger = n_bigger + 1;
    }
    n_var_sim(i - burnin_num) = arma::var(n_sim);
    n_mean_sim(i - burnin_num) = arma::mean(n_sim);
    if(i == (n_MCMC - 1)){
      X_sim_out = X_sim;
      n_sim_out = n_sim;
    }
    
  }
  double p_val = n_bigger / (double) (D_obs.n_elem);
  int mean_spike_count_extreme = 0;
  int var_spike_count_extreme = 0;
  double mean_mean_SC = arma::mean(n_mean_sim);
  double mean_var_SC = arma::mean(n_var_sim);
  double obs_diff_mean_SC = std::abs(arma::mean(n) - mean_mean_SC);
  double obs_diff_var_SC = std::abs(arma::var(n) - mean_var_SC);
  for(arma::uword i = 0; i < n_mean_sim.n_elem; i++){
    if(std::abs(mean_mean_SC - n_mean_sim(i)) > obs_diff_mean_SC){
      mean_spike_count_extreme = mean_spike_count_extreme + 1;
    }
    if(std::abs(mean_var_SC - n_var_sim(i)) > obs_diff_var_SC){
      var_spike_count_extreme = var_spike_count_extreme + 1;
    }
  }
  
  double p_val_mean_spike_count = mean_spike_count_extreme / (double) (D_obs.n_elem);
  double p_val_var_spike_count = var_spike_count_extreme / (double) (D_obs.n_elem);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("p_val", p_val),
                                         Rcpp::Named("p_val_mean_SC", p_val_mean_spike_count),
                                         Rcpp::Named("p_val_var_SC", p_val_var_spike_count));
  return output;
}



inline Rcpp::List calc_WAIC_IGP_WTA(const arma::field<arma::vec> X_A,
                                    const arma::field<arma::vec> X_B,
                                    const arma::vec n_A,
                                    const arma::vec n_B,
                                    const double& end_time,
                                    const arma::mat theta_A,
                                    const arma::mat basis_coef_A,
                                    const arma::mat theta_B,
                                    const arma::mat basis_coef_B,
                                    const arma::field<arma::mat> basis_funct_A,
                                    const arma::field<arma::mat> basis_funct_B,
                                    const double burnin_prop){
  int n_MCMC_A = theta_A.n_rows;
  int burnin_num_A = n_MCMC_A - std::floor((1 - burnin_prop) * n_MCMC_A);
  int n_MCMC_B = theta_B.n_rows;
  int burnin_num_B = n_MCMC_B - std::floor((1 - burnin_prop) * n_MCMC_B);
  
  // Placeholder for log-likelihood by observation
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  
  // calculate log-likelihood for A
  for(arma::uword i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(arma::uword i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  
  for(int i = 0; i < n_MCMC_A - burnin_num_A; i++){
    for(arma::uword j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_B - burnin_num_B; i++){
    for(arma::uword j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
  }
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
      lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
      lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
  double pwaic = 0;
  for(arma::uword i = 0; i < n_A.n_elem; i++){
      pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(arma::uword i = 0; i < n_B.n_elem; i++){
      pwaic = pwaic + arma::var(llik_B_obs.col(i));
  }
  
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (lppd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
                                          Rcpp::Named("lppd", lppd),
                                          Rcpp::Named("Effective_pars", pwaic),
                                          Rcpp::Named("llik_A", llik_A),
                                          Rcpp::Named("llik_B", llik_B));
  return output1;
}

}

#endif