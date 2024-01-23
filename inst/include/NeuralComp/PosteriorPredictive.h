#ifndef POSTERIOR_PREDICTIVE_H
#define POSTERIOR_PREDICTIVE_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include <splines2Armadillo.h>

namespace NeuralComp {

inline Rcpp::List posterior_pred_samples(const arma::mat& theta,
                                         const arma::mat& basis_coef_A,
                                         const arma::mat& basis_coef_B,
                                         const int& basis_degree,
                                         const arma::vec& boundary_knots,
                                         const arma::vec& internal_knots,
                                         const double burnin_prop,
                                         const double trial_time,
                                         const bool time_inhomogeneous){
  int n_MCMC = theta.n_rows;
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  arma::field<arma::vec> posterior_predictive_samples_AB(n_MCMC - burnin_num, 1);
  arma::field<arma::vec> posterior_predictive_labels(n_MCMC - burnin_num, 1);
  arma::vec n_AB_posterior_predictive = arma::zeros(n_MCMC - burnin_num);
  arma::vec n_switches = arma::zeros(n_MCMC - burnin_num);
  arma::field<arma::vec> switch_times(n_MCMC - burnin_num, 1);
  arma::field<arma::vec> switch_states(n_MCMC - burnin_num, 1);
  arma::vec samples_i;
  arma::vec labels_i;
  arma::vec switch_times_i;
  arma::vec switch_states_i;
  double ISI_A;
  double ISI_B;
  double total_time;
  splines2::BSpline bspline;
  arma::vec time = arma::zeros(1);
  int j;
  
  // Sample AB process
  for(int i = burnin_num; i < n_MCMC; i++){
    j = 0;
    total_time = 0;
    samples_i = arma::vec(1);
    labels_i = arma::vec(1);
    switch_times_i = arma::vec(1);
    switch_states_i = arma::vec(1);
    time(0) = total_time;
    
    if(time_inhomogeneous == true){
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      arma::mat bspline_mat{bspline.basis(true)};
      ISI_A = rinv_gauss((1/(theta(i,0) + arma::dot(bspline_mat.row(0), basis_coef_A.row(i)))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
      ISI_B = rinv_gauss((1/(theta(i,1) + arma::dot(bspline_mat.row(0), basis_coef_B.row(i)))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
    }else{
      ISI_A = rinv_gauss((1/(theta(i,0))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
      ISI_B = rinv_gauss((1/(theta(i,1))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
    }
    if(ISI_A < ISI_B){
      total_time = total_time + ISI_A;
      labels_i(j) = 0;
      samples_i(j) = ISI_A;
      switch_states_i(0) = 0;
      j = j + 1;
    }else{
      total_time = total_time + ISI_B;
      labels_i(j) = 1;
      samples_i(j) = ISI_B;
      switch_states_i(0) = 1;
      j = j + 1;
    }
    while(total_time < trial_time){
      time(0) = total_time;
      if(time_inhomogeneous == true){
        bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                    boundary_knots);
        arma::mat bspline_mat{bspline.basis(true)};
        if(labels_i(j - 1) == 0){
          ISI_A = rinv_gauss((1/(theta(i,0) + arma::dot(bspline_mat.row(0), basis_coef_A.row(i)))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
          ISI_B = theta(i,4) + rinv_gauss((1/(theta(i,1) + arma::dot(bspline_mat.row(0), basis_coef_B.row(i)))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
        }else{
          ISI_A = theta(i,4) + rinv_gauss((1/(theta(i,0) + arma::dot(bspline_mat.row(0), basis_coef_A.row(i)))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
          ISI_B = rinv_gauss((1/(theta(i,1) + arma::dot(bspline_mat.row(0), basis_coef_B.row(i)))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
        }
      }else{
        if(labels_i(j - 1) == 0){
          ISI_A = rinv_gauss((1/(theta(i,0))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
          ISI_B = theta(i,4) + rinv_gauss((1/(theta(i,1))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
        }else{
          ISI_A = theta(i,4) + rinv_gauss((1/(theta(i,0))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
          ISI_B = rinv_gauss((1/(theta(i,1))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
        }
      }
      if(ISI_A < ISI_B){
        total_time = total_time + ISI_A;
        labels_i.resize(j + 1);
        samples_i.resize(j + 1);
        if(labels_i(j - 1) != 0){
          n_switches(i - burnin_num) = n_switches(i - burnin_num) + 1;
          switch_states_i.resize(n_switches(i - burnin_num) + 1);
          switch_times_i.resize(n_switches(i - burnin_num) + 1);
          switch_states_i(n_switches(i - burnin_num)) = 0;
          if(n_switches(i - burnin_num) == 1){
            switch_times_i(0) = total_time - ISI_A;
          }else{
            switch_times_i(n_switches(i - burnin_num) - 1) = total_time - ISI_A - arma::accu(switch_times_i.subvec(0,n_switches(i - burnin_num) - 2));
          }
        }
        labels_i(j) = 0;
        samples_i(j) = ISI_A;
        j = j + 1;
      }else{
        total_time = total_time + ISI_B;
        labels_i.resize(j + 1);
        samples_i.resize(j + 1);
        if(labels_i(j - 1) != 1){
          n_switches(i- burnin_num) = n_switches(i - burnin_num) + 1;
          switch_states_i.resize(n_switches(i - burnin_num) + 1);
          switch_times_i.resize(n_switches(i - burnin_num) + 1);
          switch_states_i(n_switches(i - burnin_num)) = 1;
          if(n_switches(i - burnin_num) == 1){
            switch_times_i(0) = total_time - ISI_B;
          }else{
            switch_times_i(n_switches(i - burnin_num) - 1) = total_time - ISI_B - arma::accu(switch_times_i.subvec(0,n_switches(i - burnin_num) - 2));
          }
        }
        labels_i(j) = 1;
        samples_i(j) = ISI_B;
        j = j + 1;
      }
    }
    if(n_switches(i - burnin_num) == 0){
      switch_times_i(0) =  total_time;
    }else{
      switch_times_i(n_switches(i - burnin_num)) = total_time - arma::accu(switch_times_i.subvec(0, n_switches(i - burnin_num) - 1));
    }
    posterior_predictive_samples_AB(i - burnin_num, 0) = samples_i;
    posterior_predictive_labels(i - burnin_num, 0) = labels_i;
    n_AB_posterior_predictive(i - burnin_num) = j;
    switch_times(i - burnin_num, 0) = switch_times_i;
    switch_states(i - burnin_num, 0) = switch_states_i;
  }
  
  // Sample A process and B Process
  arma::field<arma::vec> posterior_predictive_samples_A(n_MCMC - burnin_num, 1);
  arma::vec n_A_posterior_predictive = arma::zeros(n_MCMC - burnin_num);
  arma::field<arma::vec> posterior_predictive_samples_B(n_MCMC - burnin_num, 1);
  arma::vec n_B_posterior_predictive = arma::zeros(n_MCMC - burnin_num);
  for(int i = burnin_num; i < n_MCMC; i++){
    j = 0;
    total_time = 0;
    samples_i = arma::vec(1);
    time(0) = total_time;

    // sample A process
    if(time_inhomogeneous == true){
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      arma::mat bspline_mat{bspline.basis(true)};
      ISI_A = rinv_gauss((1/(theta(i,0) + arma::dot(bspline_mat.row(0), basis_coef_A.row(i)))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
    }else{
      ISI_A = rinv_gauss((1/(theta(i,0))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
    }
    
    total_time = total_time + ISI_A;
    samples_i(j) = ISI_A;
    j = j + 1;
    while(total_time < trial_time){
      time(0) = total_time;
      
      if(time_inhomogeneous == true){
        bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                    boundary_knots);
        arma::mat bspline_mat{bspline.basis(true)};
        ISI_A = rinv_gauss((1/(theta(i,0) + arma::dot(bspline_mat.row(0), basis_coef_A.row(i)))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
      }else{
        ISI_A = rinv_gauss((1/(theta(i,0))), (1 / theta(i, 2)) * (1 / theta(i, 2)));
      }
      total_time = total_time + ISI_A;
      samples_i.resize(j + 1);
      samples_i(j) = ISI_A;
      j = j + 1;
    }
    posterior_predictive_samples_A(i - burnin_num, 0) = samples_i;
    n_A_posterior_predictive(i - burnin_num) = j;
    
    j = 0;
    total_time = 0;
    samples_i = arma::vec(1);
    time(0) = total_time;
    
    // sample B process
    if(time_inhomogeneous == true){
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      arma::mat bspline_mat1{bspline.basis(true)};
      ISI_B = rinv_gauss((1/(theta(i,1) + arma::dot(bspline_mat1.row(0), basis_coef_B.row(i)))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
    }else{
      ISI_B = rinv_gauss((1/(theta(i,1))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
    }
    total_time = total_time + ISI_B;
    samples_i(j) = ISI_B;
    j = j + 1;
    while(total_time < trial_time){
      time(0) = total_time;
      
      if(time_inhomogeneous == true){
        bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                    boundary_knots);
        arma::mat bspline_mat1{bspline.basis(true)};
        ISI_B = rinv_gauss((1/(theta(i,1) + arma::dot(bspline_mat1.row(0), basis_coef_B.row(i)))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
      }else{
        ISI_B = rinv_gauss((1/(theta(i,1))), (1 / theta(i, 3)) * (1 / theta(i, 3)));
      }
      total_time = total_time + ISI_B;
      samples_i.resize(j + 1);
      samples_i(j) = ISI_B;
      j = j + 1;
    }
    posterior_predictive_samples_B(i - burnin_num, 0) = samples_i;
    n_B_posterior_predictive(i - burnin_num) = j;
  }
  
  
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("posterior_pred_samples_A", posterior_predictive_samples_A),
                                         Rcpp::Named("posterior_pred_samples_B", posterior_predictive_samples_B),
                                         Rcpp::Named("posterior_pred_samples_AB", posterior_predictive_samples_AB),
                                         Rcpp::Named("posterior_pred_labels", posterior_predictive_labels),
                                         Rcpp::Named("n_A", n_A_posterior_predictive),
                                         Rcpp::Named("n_B", n_B_posterior_predictive),
                                         Rcpp::Named("n_AB", n_AB_posterior_predictive),
                                         Rcpp::Named("switch_times", switch_times),
                                         Rcpp::Named("switch_states", switch_states),
                                         Rcpp::Named("n_switches", n_switches));
  
  return params;
}

}

#endif
