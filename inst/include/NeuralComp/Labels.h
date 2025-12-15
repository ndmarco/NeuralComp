#ifndef NeuralComp_Labels_H
#define NeuralComp_Labels_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include "Posterior.h"

namespace NeuralComp {

inline double prob_transition_TI(double label,
                                 double label_next,
                                 const arma::vec& X_AB,
                                 arma::vec& theta,
                                 const arma::vec& basis_coef_A,
                                 const arma::vec& basis_coef_B,
                                 const arma::mat& basis_funct_AB,
                                 int spike_num,
                                 const double& end_time){
  double log_p = 0;
  
  if(label_next == 0){
    if(label == 0){
      log_p = pinv_gauss(X_AB(spike_num) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)) +
        dinv_gauss(X_AB(spike_num), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A)))), std::pow((1 / theta(2)), 2.0));
    }else{
      log_p = pinv_gauss(X_AB(spike_num), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)) +
        dinv_gauss(X_AB(spike_num) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A)))), std::pow((1 / theta(2)), 2.0));
    }
    if(spike_num == (X_AB.n_elem - 1)){
      log_p = log_p + (pinv_gauss(end_time - arma::accu(X_AB), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num + 1), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)) +
        pinv_gauss(end_time - arma::accu(X_AB) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num + 1), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)));
    }
  }else{
    if(label == 0){
      log_p = pinv_gauss(X_AB(spike_num), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)) +
        dinv_gauss(X_AB(spike_num) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B)))), std::pow((1 / theta(3)), 2.0));
    }else{
      log_p = pinv_gauss(X_AB(spike_num) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)) +
        dinv_gauss(X_AB(spike_num), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B)))), std::pow((1 / theta(3)), 2.0));
    }
    if(spike_num == (X_AB.n_elem - 1)){
      log_p = log_p + (pinv_gauss(end_time - arma::accu(X_AB), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(spike_num + 1), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)) +
        pinv_gauss(end_time - arma::accu(X_AB) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(spike_num + 1), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)));
    }
  }
  
  
  
  return std::exp(log_p);
}

inline double update_prob_TI(double Labels,
                             const arma::vec& X_AB,
                             arma::vec& theta,
                             const arma::vec& basis_coef_A,
                             const arma::vec& basis_coef_B,
                             const arma::mat& basis_funct_AB,
                             int spike_num,
                             arma::vec Prob_prev,
                             const double& end_time){
  
  double p_prev_A = Prob_prev(0) * prob_transition_TI(0, Labels, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, spike_num, end_time);
  double p_prev_B = Prob_prev(1) * prob_transition_TI(1, Labels, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, spike_num, end_time);
  
  double p = p_prev_A + p_prev_B;
  
  return p;
}


inline arma::vec backward_sim_TI(arma::mat& Prob_mat,
                                 arma::vec& theta,
                                 const arma::vec& basis_coef_A,
                                 const arma::vec& basis_coef_B,
                                 const arma::mat& basis_funct_AB,
                                 const arma::vec& X_AB,
                                 double& prob_propose, 
                                 const double& end_time){
  arma::vec prop_labels(X_AB.n_elem, arma::fill::zeros);
  
  if(R::runif(0,1) < Prob_mat(X_AB.n_elem - 1,0)){
    prop_labels(X_AB.n_elem - 1) = 0;
    prob_propose = std::log(Prob_mat(X_AB.n_elem - 1,0));
  }else{
    prop_labels(X_AB.n_elem - 1) = 1;
    prob_propose = std::log(1 - Prob_mat(X_AB.n_elem - 1,0));
  }
  
  double numerator = 0;
  double denominator = 0;
  double prob_accept_A = 0;
  
  for(int i = X_AB.n_elem - 2; i > -1; i--){
    if(prop_labels(i + 1) == 0){
      numerator = Prob_mat(i,0) * prob_transition_TI(0,0, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time);
      denominator = numerator + Prob_mat(i,1) * prob_transition_TI(1,0, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time);
    }else{
      numerator = Prob_mat(i,0) * prob_transition_TI(0,1, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time);
      denominator = numerator + Prob_mat(i,1) * prob_transition_TI(1,1, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time);
    }
    
    prob_accept_A = numerator / denominator;
    if(R::runif(0,1) < prob_accept_A){
      prop_labels(i) = 0;
      prob_propose = prob_propose + std::log(prob_accept_A);
    }else{
      prop_labels(i) = 1;
      prob_propose = prob_propose + std::log(1 - prob_accept_A);
    }
  }
  
  return prop_labels;
}


inline arma::field<arma::vec> prior_Labels(const arma::vec& n_AB,
                                           arma::mat trans_prob_0,
                                           arma::mat trans_prob){
  arma::field<arma::vec> prior_prob_labels(n_AB.n_elem, 1);
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    prior_prob_labels(i,0) = arma::zeros(n_AB(i));
    prior_prob_labels(i,0)(0) = trans_prob_0(1, 1);
    for(int j = 1; j < n_AB(i); j++){
      prior_prob_labels(i,0)(j) = (prior_prob_labels(i,0)(j-1) * trans_prob(1,1)) + 
        ((1 - prior_prob_labels(i,0)(j-1)) * trans_prob(0,1));
    }
  }
  
  return prior_prob_labels;
}



inline double calc_log_sum(arma::vec x){
  double max_val = x.max();
  
  double inner_sum = 0;
  for(arma::uword i = 0; i < x.n_elem; i++){
    inner_sum = inner_sum + std::exp(x(i) - max_val);
  }
  double output = max_val + std::log(inner_sum);
  return output;
}


inline double log_prop_q(double delta,
                         double delta_proposal_mean,
                         double delta_proposal_sd,
                         const double& delta_shape,
                         const double& delta_rate,
                         double alpha){
  double l_prob = alpha * R::dgamma(delta, delta_shape, (1 / delta_rate), false);
  l_prob = l_prob + (1 - alpha) * R::dlnorm(delta, delta_proposal_mean, delta_proposal_sd, false);
  return std::log(l_prob);
}


inline arma::field<arma::mat> forward_filtration_delta_int(arma::vec& theta,
                                                           const arma::vec& basis_coef_A,
                                                           const arma::vec& basis_coef_B,
                                                           const arma::mat& basis_funct_AB,
                                                           const arma::vec& X_AB,
                                                           const double& end_time){
  arma::mat Prob_mat(X_AB.n_elem, 2, arma::fill::zeros);
  arma::mat Prob_mat_normalization(X_AB.n_elem, 1, arma::fill::zeros);

  // Initialize probability of initial state in A or B
  Prob_mat(0,0) = std::exp(pinv_gauss(X_AB(0), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)) +
    dinv_gauss(X_AB(0), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)));
  Prob_mat(0,1) = std::exp(pinv_gauss(X_AB(0), (1 / (theta(0) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_A)))), std::pow((1 / theta(2)), 2.0)) +
    dinv_gauss(X_AB(0), (1 / (theta(1) * std::exp(arma::dot(basis_funct_AB.row(0), basis_coef_B)))), std::pow((1 / theta(3)), 2.0)));
  Prob_mat_normalization(0,0) = (Prob_mat(0,0) + Prob_mat(0,1));
  Prob_mat(0,0) = Prob_mat(0,0) / Prob_mat_normalization(0,0);
  Prob_mat(0,1) =  1 - Prob_mat(0,0);

  double numerator = 0.0;
  double denominator = 0.0;
  // Forward Pass
  for(arma::uword i = 1; i < X_AB.n_elem; i++){
    numerator = update_prob_TI(0, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i, Prob_mat.row(i-1).t(), end_time);
    
    denominator = update_prob_TI(1, X_AB, theta, basis_coef_A, basis_coef_B, basis_funct_AB, i, Prob_mat.row(i-1).t(), end_time);
    denominator = denominator + numerator;
    Prob_mat_normalization(i,0) = denominator;
    Prob_mat(i, 0) = numerator / denominator;
    Prob_mat(i, 1) = 1 - Prob_mat(i, 0);
  }
  Prob_mat_normalization = arma::log(Prob_mat_normalization);
  arma::field<arma::mat> output(2,1);
  output(0,0) = Prob_mat;
  output(1,0) = Prob_mat_normalization;
  return output;
}


inline double prob_current_delta_int(arma::vec& Labels,
                                     arma::mat& Prob_mat,
                                     arma::vec& delta,
                                     arma::vec& theta,
                                     const arma::vec& basis_coef_A,
                                     const arma::vec& basis_coef_B,
                                     const arma::mat& basis_funct_AB,
                                     const arma::vec& X_AB,
                                     double delta_proposal_mean,
                                     double delta_proposal_sd,
                                     const double& delta_shape,
                                     const double& delta_rate,
                                     const double alpha, 
                                     const double& end_time){
  double prob_state = 0;
  if(Labels(Labels.n_elem - 1) == 0){
    prob_state = std::log(Prob_mat(X_AB.n_elem - 1,0));
  }else{
    prob_state = std::log(1 - Prob_mat(X_AB.n_elem - 1,0));
  }
  
  double numerator = 0;
  double denominator = 0;
  double prob_accept_A = 0;
  double ph = 0;
  arma::vec theta_j = theta;
  for(int i = X_AB.n_elem - 2; i > -1; i--){
    if(Labels(i + 1) == 0){
      ph = 0;
      for(arma::uword j = 0; j < delta.n_elem; j++){
        theta_j(4) = delta(j);
        ph = ph + (prob_transition_TI(0,0, X_AB, theta_j, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time) / std::exp(log_prop_q(delta(j), delta_proposal_mean, delta_proposal_sd,
                                                                       delta_shape, delta_rate, alpha)));
      }
      numerator = Prob_mat(i,0) * ph;
      
      ph = 0;
      for(arma::uword j = 0; j < delta.n_elem; j++){
        theta_j(4) = delta(j);
        ph = ph + (prob_transition_TI(1,0, X_AB, theta_j, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time) / std::exp(log_prop_q(delta(j), delta_proposal_mean, delta_proposal_sd,
                                                                       delta_shape, delta_rate, alpha)));
      }
      denominator = numerator + Prob_mat(i,1) * ph;
    }else{
      ph = 0;
      for(arma::uword j = 0; j < delta.n_elem; j++){
        theta_j(4) = delta(j);
        ph = ph + (prob_transition_TI(0,1, X_AB, theta_j, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time) / std::exp(log_prop_q(delta(j), delta_proposal_mean, delta_proposal_sd,
                                                                       delta_shape, delta_rate, alpha)));
      }
      numerator = Prob_mat(i,0) * ph;
      
      ph = 0;
      for(int j = 0; j < delta.n_elem; j++){
        theta_j(4) = delta(j);
        ph = ph + (prob_transition_TI(1,1, X_AB, theta_j, basis_coef_A, basis_coef_B, basis_funct_AB, i+1, end_time) / std::exp(log_prop_q(delta(j), delta_proposal_mean, delta_proposal_sd,
                                                                       delta_shape, delta_rate, alpha)));
      }
      denominator = numerator + Prob_mat(i,1) * ph;
    }
    
    prob_accept_A = numerator / denominator;
    if(Labels(i) == 0){
      prob_state = prob_state + std::log(prob_accept_A);
    }else{
      prob_state = prob_state + std::log(1 - prob_accept_A);
    }
  }
  
  return prob_state;
}



inline void FFBS_ensemble_step1(arma::field<arma::vec>& Labels,
                                int iter,
                                const arma::field<arma::vec>& X_AB,
                                const arma::vec& n_AB,
                                arma::vec& theta,
                                const arma::vec& basis_coef_A,
                                const arma::vec& basis_coef_B,
                                const arma::field<arma::mat>& basis_funct_AB,
                                double delta_proposal_mean,
                                double delta_proposal_sd,
                                const double alpha,
                                int M_proposal,
                                const double& delta_shape,
                                const double& delta_rate,
                                const double& end_time){
  double prob_propose = 0;
  
  arma::field<arma::vec> Labels_ensembles_ph(n_AB.n_elem, 1);
  arma::vec theta_exp = arma::exp(theta);
  arma::vec theta_0 = theta_exp;
  theta_0(4) = 0;
  arma::vec delta_ensemble = arma::zeros(M_proposal);
  // Set initial positions of delta ensemble
  delta_ensemble(0) = theta_exp(4);
  for(int i = 1; i < M_proposal; i++){
    if(R::runif(0,1) < alpha){
      delta_ensemble(i) = R::rgamma(delta_shape, (1 / delta_rate));
    }else{
      delta_ensemble(i) = R::rlnorm(delta_proposal_mean, delta_proposal_sd);
    }
  }
  arma::vec ph2 = arma::zeros(2);
  
  arma::field<arma::mat> forward_filtrations_delta(M_proposal, n_AB.n_elem);
  arma::vec theta_j = theta_exp;
  arma::vec rel_probs = arma::zeros(M_proposal);
  // Start sampling delta
  for(int j = 0; j < M_proposal; j++){
    theta_j(4) = delta_ensemble(j);
    for(arma::uword i = 0; i < n_AB.n_elem; i++){
      if(n_AB(i) > 0){
        arma::field<arma::mat> output = forward_filtration_delta_int(theta_j, basis_coef_A, basis_coef_B, basis_funct_AB(i,0), X_AB(i, 0), end_time);
        forward_filtrations_delta(j, i) = output(0, 0);
        rel_probs(j) = rel_probs(j) +  arma::accu(output(1, 0));
      }
    }
    rel_probs(j) = rel_probs(j) + R::dgamma(delta_ensemble(j), delta_shape, (1 / delta_rate), true) - 
      log_prop_q(delta_ensemble(j), delta_proposal_mean, delta_proposal_sd, delta_shape, delta_rate, alpha);
  }
  arma::vec probs = arma::zeros(M_proposal);
  for(int i = 0; i < M_proposal; i++){
    probs(i) = exp(rel_probs(i) - calc_log_sum(rel_probs));
  }
  arma::vec draw = rmutlinomial(probs);
  int delta_index = 0;
  // Update delta
  for(arma::uword i = 0; i < draw.n_elem; i++){
    if(draw(i) == 1){
      theta(4) = std::log(delta_ensemble(i));
      delta_index = i;
    }
  }
  
  //update theta
  theta_exp = arma::exp(theta);
  // Start sampling the labels
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    if(n_AB(i) > 0){
      arma::mat Prob_mat = forward_filtrations_delta(delta_index, i);
      arma::vec proposed_labels = backward_sim_TI(Prob_mat, theta_exp, basis_coef_A, 
                                                  basis_coef_B, basis_funct_AB(i, 0), 
                                                  X_AB(i, 0), prob_propose, end_time);
      
      Labels(i, iter) = proposed_labels;
    }
  }
}


inline void FFBS_delta_seperate(arma::field<arma::vec>& Labels,
                                int iter,
                                const arma::field<arma::vec>& X_A,
                                const arma::field<arma::vec>& X_B,
                                const arma::field<arma::vec>& X_AB,
                                const arma::vec& n_A,
                                const arma::vec& n_B,
                                const arma::vec& n_AB,
                                arma::vec& theta,
                                arma::vec basis_coef_A,
                                arma::vec basis_coef_B,
                                const arma::field<arma::mat>& basis_funct_A,
                                const arma::field<arma::mat>& basis_funct_B,
                                const arma::field<arma::mat>& basis_funct_AB,
                                const double& delta_sigma,
                                const double& delta_shape,
                                const double& delta_rate,
                                const double& end_time){

  arma::field<arma::vec> Labels_ensembles_ph(n_AB.n_elem, 1);
  arma::vec theta_exp = arma::exp(theta);
  arma::vec theta_exp_propose = arma::exp(theta);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  
  for(arma::uword j = 0; j < n_AB.n_elem; j++){
    Labels_iter(j,0) = Labels(j, iter);
  }
  
  
  // Sample delta
  theta_exp_propose(4) = std::exp(std::log(theta_exp(4)) + R::rnorm(0, delta_sigma));
  double prob_current = log_likelihood_TI(Labels_iter, theta_exp, basis_coef_A, basis_coef_B,
                                          basis_funct_A, basis_funct_B,  basis_funct_AB,
                                          X_A, X_B, X_AB, n_A, n_B, n_AB, end_time) + R::dgamma(theta_exp(4), delta_shape, (1 / delta_rate), true);
  
  double prob_propose = log_likelihood_TI(Labels_iter, theta_exp_propose, basis_coef_A, basis_coef_B,
                                          basis_funct_A, basis_funct_B,  basis_funct_AB,
                                          X_A, X_B, X_AB, n_A, n_B, n_AB, end_time) + R::dgamma(theta_exp_propose(4), delta_shape, (1 / delta_rate), true);
  if(std::log(R::runif(0,1)) < (prob_propose - prob_current)){
    theta(4) = std::log(theta_exp_propose(4));
  }
  
  arma::field<arma::mat> forward_filtrations_delta(1, n_AB.n_elem);
  theta_exp = arma::exp(theta);
  // Start sampling delta
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    arma::field<arma::mat> output = forward_filtration_delta_int(theta_exp, basis_coef_A, basis_coef_B, basis_funct_AB(i,0), X_AB(i, 0), end_time);
    forward_filtrations_delta(0, i) = output(0, 0);
  }
  
  // Start sampling the labels
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    arma::mat Prob_mat = forward_filtrations_delta(0, i);
    arma::vec proposed_labels = backward_sim_TI(Prob_mat, theta_exp, basis_coef_A, 
                                                basis_coef_B, basis_funct_AB(i, 0), 
                                                X_AB(i, 0), prob_propose, end_time);
    
    Labels(i, iter) = proposed_labels;
  }
  
}


inline Rcpp::List FFBS_joint(const arma::field<arma::vec>& X_AB,
                             const arma::vec& n_AB,
                             arma::mat& theta,
                             const arma::vec& basis_coef_A,
                             const arma::vec& basis_coef_B,
                             const arma::field<arma::mat>& basis_funct_AB,
                             double delta_proposal_mean,
                             double delta_proposal_sd,
                             const double alpha,
                             int M_proposal,
                             const double& delta_shape,
                             const double& delta_rate,
                             const double& end_time,
                             int MCMC_iters,
                             int Warm_block1,
                             int Warm_block2,
                             int delta_adaption_block){

  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block1 + Warm_block2);
  // Use initial starting position
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block1 + Warm_block2; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
  }
  arma::vec theta_ph = theta.row(0).t();
  for(int i = 1; i < Warm_block1; i++){
    theta_ph = theta.row(i).t();
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A, basis_coef_B, 
                        basis_funct_AB, delta_proposal_mean, delta_proposal_sd, 
                        alpha, M_proposal, delta_shape, delta_rate, end_time);
    theta.row(i) = theta_ph.t();
    if((i + 1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(arma::uword j = 0; j < X_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if( (i % 50) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
  }
  
  double delta_proposal_meani = arma::mean(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  double delta_proposal_sdi = arma::stddev(theta.col(4).subvec(Warm_block1 - std::floor(0.5 *Warm_block1), Warm_block1 - 1));
  if(delta_proposal_sdi == 0.00){
    delta_proposal_sdi = 0.005;
  }
  
  for(int i = Warm_block1; i < Warm_block1 + Warm_block2; i++){
    theta_ph = theta.row(i).t();
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A, basis_coef_B, 
                        basis_funct_AB, delta_proposal_meani, delta_proposal_sdi, 
                        alpha, M_proposal, delta_shape, delta_rate, end_time);
    theta.row(i) = theta_ph.t();
    if((i + 1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(int j = 0; j < X_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if((i % 50) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
    if(i > Warm_block1){
      if(((i - Warm_block1) % delta_adaption_block) == 0){
        delta_proposal_meani = arma::mean(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        delta_proposal_sdi = arma::stddev(theta.col(4).subvec(i - delta_adaption_block, i - 1));
        if(delta_proposal_sdi == 0.00){
          delta_proposal_sdi = 0.005;
        }
      }
    }
  }
  

  
  for(int i = Warm_block1 + Warm_block2; i < MCMC_iters + Warm_block1 + Warm_block2; i++){
    theta_ph = theta.row(i).t();
    FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_ph, basis_coef_A, basis_coef_B, 
                        basis_funct_AB, delta_proposal_meani, delta_proposal_sdi, 
                        alpha, M_proposal, delta_shape, delta_rate, end_time);
    theta.row(i) = theta_ph.t();
    if((i + 1) < Warm_block1 + Warm_block2 + MCMC_iters){
      theta.row(i + 1) = theta.row(i);
      for(arma::uword j = 0; j < X_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if( (i % 50) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
  }
  
  //convert labels
  arma::field<arma::mat> labels_out(n_AB.n_elem, 1);
  for(arma::uword i = 0; i < n_AB.n_elem; i++){
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
                                         Rcpp::Named("basis_coef_B", basis_coef_B));
  
  return params;
}

}

#endif