#ifndef WAIC_H
#define WAIC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include <splines2Armadillo.h>

namespace NeuralComp {

inline arma::mat approximate_L_given_theta(arma::vec theta,
                                           arma::vec basis_coef_A,
                                           arma::vec basis_coef_B,
                                           arma::vec basis_func,
                                           double max_time,
                                           int n_eval){
  arma::mat P_mat(2, 2, arma::fill::zeros);
  arma::vec eval = arma::linspace(0, max_time, n_eval);
  double step_size = eval(1) - eval(0);
  
  for(int i = 0; i < n_eval; i++){
    P_mat(0,0) = P_mat(0,0) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)) +
      dinv_gauss(eval(i), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2))) * step_size;
    P_mat(1,1) = P_mat(1,1) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)) +
      dinv_gauss(eval(i), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2))) * step_size;
  }
  
  eval = arma::linspace(theta(4), max_time + theta(4), n_eval);
  for(int i = 0; i < n_eval; i++){
    P_mat(0,1) = P_mat(0,1) + std::exp(pinv_gauss(eval(i), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)) +
      dinv_gauss(eval(i) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2))) * step_size;
    P_mat(1,0) = P_mat(1,0) + std::exp(pinv_gauss(eval(i), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)) +
      dinv_gauss(eval(i) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2))) * step_size;
  }
  
  P_mat(0,0) = (P_mat(0,0) + (1 - P_mat(0,1))) / 2;
  P_mat(0,1) = 1 - P_mat(0,0);
  P_mat(1,0) = (P_mat(1,0) + (1 - P_mat(1,1))) / 2;
  P_mat(1,1) = 1 - P_mat(1,0);
  
  return P_mat;
}

inline double joint_prob00(arma::vec theta,
                         arma::vec basis_coef_A,
                         arma::vec basis_coef_B,
                         arma::vec basis_func,
                         double x){
  double output = std::exp(pinv_gauss(x - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)) +
    dinv_gauss(x, (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)));
  
  return output;
}

inline double joint_prob01(arma::vec theta,
                           arma::vec basis_coef_A,
                           arma::vec basis_coef_B,
                           arma::vec basis_func,
                           double x){
  double output = std::exp(pinv_gauss(x, (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)) +
                           dinv_gauss(x - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)));
  
  return output;
}

inline double joint_prob10(arma::vec theta,
                           arma::vec basis_coef_A,
                           arma::vec basis_coef_B,
                           arma::vec basis_func,
                           double x){
  double output = std::exp(pinv_gauss(x, (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)) +
                        dinv_gauss(x - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)));
  
  return output;
}

inline double joint_prob11(arma::vec theta,
                           arma::vec basis_coef_A,
                           arma::vec basis_coef_B,
                           arma::vec basis_func,
                           double x){
  double output = std::exp(pinv_gauss(x - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), pow((1 / theta(2)), 2)) +
                           dinv_gauss(x, (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), pow((1 / theta(3)), 2)));
  
  return output;
}

inline arma::mat approximate_L_given_theta_simpson(arma::vec theta,
                                                   arma::vec basis_coef_A,
                                                   arma::vec basis_coef_B,
                                                   arma::vec basis_func,
                                                   double max_time,
                                                   int n_eval){
  arma::mat P_mat(2, 2, arma::fill::zeros);
  arma::vec eval = arma::linspace(0, max_time, (3 * n_eval) + 1);
  double step_size = eval(1) - eval(0);
  
  double ph = 0;
  for(int i = 0; i < eval.n_elem; i++){
    if(i == 0){
      ph = ph + joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if(i == (eval.n_elem - 1)){
      ph = ph + joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if((i % 3) != 0){
      ph = ph + 3 * joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else{
      ph = ph + 2 * joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }
  }
  P_mat(0,0) = (3 * step_size * ph) / 8;
  
  ph = 0;
  for(int i = 0; i < eval.n_elem; i++){
    if(i == 0){
      ph = ph + joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if(i == (eval.n_elem - 1)){
      ph = ph + joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if((i % 3) != 0){
      ph = ph + 3 * joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else{
      ph = ph + 2 * joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }
  }
  P_mat(0,1) = (3 * step_size * ph) / 8;
  
  ph = 0;
  for(int i = 0; i < eval.n_elem; i++){
    if(i == 0){
      ph = ph + joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if(i == (eval.n_elem - 1)){
      ph = ph + joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if((i % 3) != 0){
      ph = ph + 3 * joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else{
      ph = ph + 2 * joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }
  }
  P_mat(1,0) = (3 * step_size * ph) / 8;
  
  ph = 0;
  for(int i = 0; i < eval.n_elem; i++){
    if(i == 0){
      ph = ph + joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if(i == (eval.n_elem - 1)){
      ph = ph + joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else if((i % 3) != 0){
      ph = ph + 3 * joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }else{
      ph = ph + 2 * joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
    }
  }
  P_mat(1,1) = (3 * step_size * ph) / 8;
  if(((P_mat(0,0) + (1 - P_mat(0,1))) / 2) > 0){
    P_mat(0,0) = (P_mat(0,0) + (1 - P_mat(0,1))) / 2;
  }
  P_mat(0,1) = 1 - P_mat(0,0);
  if(((P_mat(1,0) + (1 - P_mat(1,1))) / 2) > 0){
    P_mat(1,0) = (P_mat(1,0) + (1 - P_mat(1,1))) / 2;
  }
  P_mat(1,1) = 1 - P_mat(1,0);
  
  return P_mat;
}


inline arma::mat calc_loglikelihood_A(const arma::vec X_A,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_A,
                                      const arma::mat basis_funct_A,
                                      const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik = arma::zeros(n_MCMC - burnin_num, X_A.n_elem);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X_A.n_elem; j++){
      llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X_A(j), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_A.row(j), basis_coef_A.row(i))))),
           pow((1 / theta(i, 2)), 2));
    }
  }
  
  return llik;
}

inline arma::mat calc_loglikelihood_B(const arma::vec X_B,
                                      const arma::mat theta,
                                      const arma::mat basis_coef_B,
                                      const arma::mat basis_funct_B,
                                      const double burnin_prop){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik = arma::zeros(n_MCMC - burnin_num, X_B.n_elem);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X_B.n_elem; j++){
      llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X_B(j), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_B.row(j), basis_coef_B.row(i))))),
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
                                       const int obs_num,
                                       const int spike_num,
                                       const double burnin_prop,
                                       const int P_mat_index){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::vec llik = arma::zeros(n_MCMC - burnin_num);
  
  for(int i = burnin_num; i < n_MCMC; i++){
    if(Labels(obs_num, 0)(i,spike_num) == 0){
      // label is A
      if(spike_num != 0){
        if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
          // Condition if spike has not switched (still in A)
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
                                                         pow((1 / theta(i, 3)), 2)) +
                                                           dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(0, 0)));
        }else{
          // Condition if spike has switched from B to A
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), pow((1 / theta(i, 3)), 2)) +
            dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(1, 0)));
        }
      }else{
        llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), pow((1 / theta(i, 3)), 2)) +
          dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(0, 0)));
      }
    }else{
      // label is B
      if(spike_num != 0){
        if(Labels(obs_num, 0)(i, spike_num-1) == 1){
          // Condition if spike has not switched (still in B)
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2)) +
            dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(1, 1)));
        }else{
          // Condition if spike has switched from A to B
          llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2)) +
            dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(0, 1)));
        }
      }else{
        llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), pow((1 / theta(i, 2)), 2)) +
          dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), pow((1 / theta(i, 3)), 2))) - std::log(P_mat(i - burnin_num, P_mat_index)(1, 1)));
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
                                    const double max_spike_time,
                                    const int n_spikes_eval,
                                    const int n_eval,
                                    const int basis_degree,
                                    const arma::vec boundary_knots,
                                    const arma::vec internal_knots){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  
  // Placeholder for log-likelihood by observation
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  
  arma::field<arma::mat> P_mat0(n_MCMC - burnin_num, 1);
  arma::mat theta_0 = theta;
  theta_0.col(4) = arma::zeros(theta_0.n_rows);
  
  for(int i = burnin_num; i < n_MCMC; i++){
    P_mat0(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta_0.row(i).t(), basis_coef_A.row(i).t(),
           basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
  }
  
  arma::field<arma::mat> P_mat(n_MCMC - burnin_num, n_spikes_eval);
  arma::vec time_vec;
  if(max_spike_time != 0){
    time_vec = arma::linspace(0, max_spike_time, n_spikes_eval);
    splines2::BSpline bspline;
    bspline = splines2::BSpline(time_vec, internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix
    arma::mat bspline_mat{bspline.basis(false)};
    for(int j = 0; j < n_spikes_eval; j++){
      for(int i = burnin_num; i < n_MCMC; i++){
        P_mat(i - burnin_num, j) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
               basis_coef_B.row(i).t(), bspline_mat.row(j).t(), max_time, n_eval);
      }
      Rcpp::Rcout << "Approximated transition probability for spike " << j + 1 << " out of " << n_spikes_eval << "\n";
    }
  }else{
    for(int i = burnin_num; i < n_MCMC; i++){
      P_mat(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
             basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
    }
  }
  
  
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
  }
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for AB
  int index;
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(max_spike_time != 0){
        if(j == 0){
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0);
        }else{
          index = arma::index_min(arma::square(time_vec - arma::accu(X_AB(i,0).subvec(0,j-1))));
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, index);
        }
      }else{
        if(j == 0){
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0);
        }
        else{
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, 0);
        }
      }
      
    }
    Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
    }
  }
  
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
    }
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      pwaic = pwaic + arma::var(llik_A(i,0).col(j));
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      pwaic = pwaic + arma::var(llik_B(i,0).col(j));
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
    }
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}


inline double calc_WAIC_competition_observation(const arma::field<arma::vec> X_A,
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
                                                const double max_spike_time,
                                                const int n_spikes_eval,
                                                const int n_eval,
                                                const int basis_degree,
                                                const arma::vec boundary_knots,
                                                const arma::vec internal_knots){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  
  // Placeholder for log-likelihood by observation
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  
  arma::field<arma::mat> P_mat0(n_MCMC - burnin_num, 1);
  arma::mat theta_0 = theta;
  theta_0.col(4) = arma::zeros(theta_0.n_rows);
  
  for(int i = burnin_num; i < n_MCMC; i++){
    P_mat0(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta_0.row(i).t(), basis_coef_A.row(i).t(),
           basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
  }
  
  arma::field<arma::mat> P_mat(n_MCMC - burnin_num, n_spikes_eval);
  arma::vec time_vec;
  if(max_spike_time != 0){
    time_vec = arma::linspace(0, max_spike_time, n_spikes_eval);
    splines2::BSpline bspline;
    bspline = splines2::BSpline(time_vec, internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix
    arma::mat bspline_mat{bspline.basis(false)};
    for(int j = 0; j < n_spikes_eval; j++){
      for(int i = burnin_num; i < n_MCMC; i++){
        P_mat(i - burnin_num, j) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
              basis_coef_B.row(i).t(), bspline_mat.row(j).t(), max_time, n_eval);
      }
      Rcpp::Rcout << "Approximated transition probability for spike " << j + 1 << " out of " << n_spikes_eval << "\n";
    }
  }else{
    for(int i = burnin_num; i < n_MCMC; i++){
      P_mat(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
            basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
    }
  }
  
  
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
  }
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for AB
  int index;
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(max_spike_time != 0){
        if(j == 0){
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0);
        }else{
          index = arma::index_min(arma::square(time_vec - arma::accu(X_AB(i,0).subvec(0,j-1))));
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, index);
        }
      }else{
        if(j == 0){
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0);
        }
        else{
          llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
                  basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, 0);
        }
      }
      
    }
    Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC - burnin_num, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC - burnin_num, n_B.n_elem);
  arma::mat llik_AB_obs = arma::zeros(n_MCMC - burnin_num, n_AB.n_elem);
  
  for(int i = 0; i < n_MCMC - burnin_num; i++){
    for(int j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
    for(int j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
    for(int j = 0; j < n_AB.n_elem; j++){
      llik_AB_obs(i, j) = arma::accu(llik_AB(j,0).row(i));
    }
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_A_obs.col(i))));
  }
  
  for(int i = 0; i < n_B.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_B_obs.col(i))));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_AB_obs.col(i))));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B_obs.col(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}

inline arma::mat calc_loglikelihood_IGP(const arma::vec X,
                                        const arma::mat theta,
                                        const arma::mat basis_coef,
                                        const arma::mat basis_funct,
                                        const double burnin_prop){
  
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::mat llik = arma::zeros(n_MCMC - burnin_num, X.n_elem);
  for(int i = burnin_num; i < n_MCMC; i++){
    for(int j = 0; j < X.n_elem; j++){
      llik(i - burnin_num, j) = llik(i - burnin_num, j) + dinv_gauss(X(j), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct.row(j), basis_coef.row(i))))),
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
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
                basis_funct_AB(i,0), burnin_prop);
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      llpd = llpd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
    }
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      pwaic = pwaic + arma::var(llik_A(i,0).col(j));
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      pwaic = pwaic + arma::var(llik_B(i,0).col(j));
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
    }
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}

inline double calc_WAIC_IGP_observation(const arma::field<arma::vec> X_A,
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
  arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
  arma::field<arma::mat> llik_B(n_B.n_elem, 1);
  arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop);
  }
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
            basis_funct_AB(i,0), burnin_prop);
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  arma::mat llik_AB_obs = arma::zeros(n_MCMC_AB - burnin_num_AB, n_AB.n_elem);
  
  for(int i = 0; i < n_MCMC_A - burnin_num_A; i++){
    for(int j = 0; j < n_A.n_elem; j++){
      llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_B - burnin_num_B; i++){
    for(int j = 0; j < n_B.n_elem; j++){
      llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
    }
  }
  
  for(int i = 0; i < n_MCMC_AB - burnin_num_AB; i++){
    for(int j = 0; j < n_AB.n_elem; j++){
      llik_AB_obs(i, j) = arma::accu(llik_AB(j,0).row(i));
    }
  }
  
  // calculate log pointwise predictive density
  double llpd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_A_obs.col(i))));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_B_obs.col(i))));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    llpd = llpd + std::log(arma::mean(arma::exp(llik_AB_obs.col(i))));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << llpd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    pwaic = pwaic + arma::var(llik_B_obs.col(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    pwaic = pwaic + arma::var(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
  double waic = -2 * (llpd - pwaic);
  
  Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
  
  return waic;
}


}


#endif