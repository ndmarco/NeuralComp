#ifndef NeuralComp_Labels_H
#define NeuralComp_Labels_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include "Posterior.h"

namespace NeuralComp {

inline double posterior_Z(arma::vec& Labels,
                          const arma::vec& X_AB,
                          arma::vec& theta,
                          int spike_num,
                          int n_AB){
  double log_p = 0;
  
  if(Labels(spike_num) == 0){
    // label is A
    if(spike_num != 0){
      if(Labels(spike_num-1) == 0){
        // Condition if spike has not switched (still in A)
        log_p = log_p + pinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
          dinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2));
      }else{
        // Condition if spike has switched from B to A
        log_p = log_p + pinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2)) +
          dinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
      }
    }else{
      log_p = log_p + pinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2)) +
        dinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2));
    }
  }else{
    // label is B
    if(spike_num != 0){
      if(Labels(spike_num-1) == 1){
        // Condition if spike has not switched (still in B)
        log_p = log_p + pinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
          dinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2));
      }else{
        // Condition if spike has switched from A to B
        log_p = log_p + pinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2)) +
          dinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
      }
    }else{
      log_p = log_p + pinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2)) +
        dinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2));
    }
  }
  
  // dependency on the next spike
  if((spike_num + 1) < n_AB){
    if(Labels(spike_num + 1) == 0){
      // label is A
      if((spike_num + 1) != 0){
        if(Labels(spike_num) == 0){
          // Condition if spike has not switched (still in A)
          log_p = log_p + pinv_gauss(X_AB(spike_num + 1) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(spike_num + 1), (1 / theta(0)), pow((1 / theta(2)), 2));
        }else{
          // Condition if spike has switched from B to A
          log_p = log_p + pinv_gauss(X_AB(spike_num + 1), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(spike_num + 1) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
        }
      }else{
        log_p = log_p + pinv_gauss(X_AB(spike_num + 1), (1 / theta(1)), pow((1 / theta(3)), 2)) +
          dinv_gauss(X_AB(spike_num + 1), (1 / theta(0)), pow((1 / theta(2)), 2));
      }
    }else{
      // label is B
      if((spike_num + 1) != 0){
        if(Labels(spike_num) == 1){
          // Condition if spike has not switched (still in B)
          log_p = log_p + pinv_gauss(X_AB(spike_num + 1) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(spike_num + 1), (1 / theta(1)), pow((1 / theta(3)), 2));
        }else{
          // Condition if spike has switched from B to A
          log_p = log_p + pinv_gauss(X_AB(spike_num + 1), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(spike_num + 1) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
        }
      }else{
        log_p = log_p + pinv_gauss(X_AB(spike_num + 1), (1 / theta(0)), pow((1 / theta(2)), 2)) +
          dinv_gauss(X_AB(spike_num + 1), (1 / theta(1)), pow((1 / theta(3)), 2));
      }
    }
  }

  
  return log_p;
}

inline void sample_labels_step(arma::field<arma::vec>& Labels,
                               int iter,
                               const arma::field<arma::vec>& X_AB,
                               const arma::vec& n_AB,
                               arma::vec& theta){
  double Z_prop = 0;
  double l_accept = 0;
  arma::field<arma::vec> Labels_prop(X_AB.n_elem, 1);
  for(int i = 0; i < X_AB.n_elem; i++){
    Labels_prop(i,0) = Labels(i, iter);
  }

  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB[i]; j++){
      for(int k = 0; k < X_AB.n_elem; k++){
        Labels_prop(k,0) = Labels(k, iter);
      }

      if(Labels_prop(i,0)(j) == 0){
        Labels_prop(i,0)(j) = 1;
      }else{
        Labels_prop(i,0)(j) = 0;
      }
      
      l_accept = posterior_Z(Labels_prop(i,0), X_AB(i,0), theta, j, n_AB(i)) - 
        posterior_Z(Labels(i, iter), X_AB(i,0), theta, j, n_AB(i));
      
      if(std::log(R::runif(0,1)) < l_accept){
        //Rcpp::Rcout << "Accepted";
        Labels(i, iter) = Labels_prop(i, 0);
      }
    }
  }
}


inline arma::field<arma::vec> labels_sampler(const arma::field<arma::vec>& X_AB,
                                             const arma::field<arma::vec>& init_position,
                                             const arma::vec& n_AB,
                                             arma::vec& theta,
                                             int MCMC_iters){
  
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters);
  
  // Use initial starting position
  for(int j = 0; j < MCMC_iters; j++){
    for(int i = 0; i < n_AB.n_elem; i++){
      Labels(i, j) = init_position(i,0);
    }
  }
  
  for(int i = 1; i < MCMC_iters; i++){
    sample_labels_step(Labels, i, X_AB, n_AB, theta);
    if((i + 1) < MCMC_iters){
      for(int j = 0; j < X_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if( (i % 50) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
  }
  return Labels;
}

inline arma::mat approx_trans_prob(double step_size,
                                   int num_evals,
                                   arma::vec& theta){
  arma::mat P_mat(2, 2, arma::fill::zeros);
  arma::vec eval = arma::linspace(0, step_size * num_evals, num_evals);
  for(int i = 0; i < num_evals; i++){
    P_mat(0,0) = P_mat(0,0) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
      dinv_gauss(eval(i), (1 / theta(0)), pow((1 / theta(2)), 2))) * step_size;
    P_mat(1,1) = P_mat(1,1) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
      dinv_gauss(eval(i), (1 / theta(1)), pow((1 / theta(3)), 2))) * step_size;
  }
  
  eval = arma::linspace(theta(4), step_size * num_evals + theta(4), num_evals);
  for(int i = 0; i < num_evals; i++){
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

inline double prob_labels(double Labels,
                          const arma::vec& X_AB,
                          arma::vec& theta,
                          int spike_num){
  double log_p = 0;
  
  if(Labels == 0){
    log_p = dinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2));
  }else{
    log_p = dinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2));
  }
  
  return std::exp(log_p);
}

inline double prob_transition(double label,
                              double label_next,
                              const arma::vec& X_AB,
                              arma::vec& theta,
                              int spike_num){
  double log_p = 0;
  
  if(label_next == 0){
    if(label == 0){
      log_p = pinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
        dinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2));
    }else{
      log_p = pinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2)) +
        dinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
    }
  }else{
    if(label == 0){
      log_p = pinv_gauss(X_AB(spike_num), (1 / theta(0)), pow((1 / theta(2)), 2)) +
        dinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
    }else{
      log_p = pinv_gauss(X_AB(spike_num) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
        dinv_gauss(X_AB(spike_num), (1 / theta(1)), pow((1 / theta(3)), 2));
    }
  }
  
  return std::exp(log_p);
}


inline arma::mat forward_pass(arma::vec& theta,
                              const arma::vec& X_AB,
                              arma::mat trans_prob_0,
                              arma::mat trans_prob){
  arma::mat Prob_mat(X_AB.n_elem, 2, arma::fill::zeros);
  
  // Initialize probability of initial state in A or B
  Prob_mat(0,0) = trans_prob_0(0,0);
  Prob_mat(0,1) = trans_prob_0(1,1);
  
  arma::vec prediction_step(2, arma::fill::zeros);
  double numerator = 0.0;
  double denominator = 0.0;
  // Forward Pass
  for(int i = 1; i < X_AB.n_elem; i++){
    // Prediction
    prediction_step = arma::zeros(2);
    prediction_step(0) = (trans_prob(0, 0) * Prob_mat(i-1, 0)) + (trans_prob(1, 0) * Prob_mat(i-1, 1));
    prediction_step(1) = (trans_prob(0, 1) * Prob_mat(i-1, 0)) + (trans_prob(1, 1) * Prob_mat(i-1, 1));
    
    // Update
    numerator = prediction_step(0) * prob_labels(0, X_AB, theta, i);
    denominator = numerator + prediction_step(1) * prob_labels(1, X_AB, theta, i);
    Prob_mat(i, 0) = numerator / denominator;
    Prob_mat(i, 1) = 1 - Prob_mat(i, 0);
  }
  
  return Prob_mat;
}


inline arma::vec backward_sim(arma::mat& Prob_mat,
                              arma::vec& theta,
                              const arma::vec& X_AB,
                              double& prob_propose){
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
      numerator = Prob_mat(i,0) * prob_transition(0,0, X_AB, theta, i+1);
      denominator = numerator + Prob_mat(i,1) * prob_transition(1,0, X_AB, theta, i+1);
    }else{
      numerator = Prob_mat(i,0) * prob_transition(0,1, X_AB, theta, i+1);
      denominator = numerator + Prob_mat(i,1) * prob_transition(1,1, X_AB, theta, i+1);
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

inline double prob_current(arma::vec& Labels,
                           arma::mat& Prob_mat,
                           arma::vec& theta,
                           const arma::vec& X_AB){
  double prob_state = 0;
  if(Labels(Labels.n_elem - 1) == 0){
    prob_state = std::log(Prob_mat(X_AB.n_elem - 1,0));
  }else{
    prob_state = std::log(1 - Prob_mat(X_AB.n_elem - 1,0));
  }
  
  double numerator = 0;
  double denominator = 0;
  double prob_accept_A = 0;
  
  for(int i = X_AB.n_elem - 2; i > -1; i--){
    if(Labels(i + 1) == 0){
      numerator = Prob_mat(i,0) * prob_transition(0,0, X_AB, theta, i+1);
      denominator = numerator + Prob_mat(i,1) * prob_transition(1,0, X_AB, theta, i+1);
    }else{
      numerator = Prob_mat(i,0) * prob_transition(0,1, X_AB, theta, i+1);
      denominator = numerator + Prob_mat(i,1) * prob_transition(1,1, X_AB, theta, i+1);
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

inline arma::field<arma::vec> prior_Labels(const arma::vec& n_AB,
                                           arma::mat trans_prob_0,
                                           arma::mat trans_prob){
  arma::field<arma::vec> prior_prob_labels(n_AB.n_elem, 1);
  for(int i = 0; i < n_AB.n_elem; i++){
    prior_prob_labels(i,0) = arma::zeros(n_AB(i));
    prior_prob_labels(i,0)(0) = trans_prob_0(1, 1);
    for(int j = 1; j < n_AB(i); j++){
      prior_prob_labels(i,0)(j) = (prior_prob_labels(i,0)(j-1) * trans_prob(1,1)) + 
        ((1 - prior_prob_labels(i,0)(j-1)) * trans_prob(0,1));
    }
  }
  
  return prior_prob_labels;
}

inline double posterior_Labels(arma::vec& Labels,
                               const arma::vec& X_AB,
                               arma::vec &theta,
                               const arma::vec& prior_p_labels){
  double posterior = 0;
  for(int j = 0; j < X_AB.n_elem; j++){
    if(Labels(j) == 0){
      // label is A
      // incorporate prior
      posterior = posterior + std::log(1 - prior_p_labels(j));
      if(j != 0){
        if(Labels(j-1) == 0){
          // Condition if spike has not switched (still in A)
          posterior = posterior + pinv_gauss(X_AB(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(j), (1 / theta(0)), pow((1 / theta(2)), 2));
        }else{
          // Condition if spike has switched from B to A
          posterior = posterior + pinv_gauss(X_AB(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            dinv_gauss(X_AB(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
        }
      }else{
        posterior = posterior + pinv_gauss(X_AB(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
          dinv_gauss(X_AB(j), (1 / theta(0)), pow((1 / theta(2)), 2));
      }
    }else{
      // label is B
      // incorporate prior
      posterior = posterior + std::log(prior_p_labels(j));
      if(j != 0){
        if(Labels(j-1) == 1){
          // Condition if spike has not switched (still in A)
          posterior = posterior + pinv_gauss(X_AB(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(j), (1 / theta(1)), pow((1 / theta(3)), 2));
        }else{
          // Condition if spike has switched from B to A
          posterior = posterior + pinv_gauss(X_AB(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            dinv_gauss(X_AB(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
        }
      }else{
        posterior = posterior + pinv_gauss(X_AB(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
          dinv_gauss(X_AB(j), (1 / theta(1)), pow((1 / theta(3)), 2));
      }
    }
  }
  
  return posterior;
}

inline double posterior_Labels_delta(arma::field<arma::vec>& Labels,
                                     const arma::field<arma::vec>& X_AB,
                                     const arma::vec& n_AB,
                                     arma::vec& theta,
                                     arma::field<arma::vec>& prior_labels,
                                     const double& delta_shape,
                                     const double& delta_rate){
  double posterior = 0;
  for(int i = 0; i < n_AB.n_elem; i++){
    posterior = posterior + posterior_Labels(Labels(i,0), X_AB(i,0), theta, prior_labels(i,0));
  }
  posterior = posterior + log_prior_delta(delta_shape, delta_rate, theta);
  return posterior;
}

inline void FFBS_step(arma::field<arma::vec>& Labels,
                      int iter,
                      const arma::field<arma::vec>& X_AB,
                      const arma::vec& n_AB,
                      arma::vec theta,
                      const double step_size,
                      const int num_evals,
                      int& accept_num){
  double prob_propose = 0;
  double prob_current1 = 0;
  double prob_accept = 0;
  arma::vec theta_exp = arma::exp(theta);
  arma::vec theta_0 = theta_exp;
  theta_0(4) = 0;
  arma::mat trans_prob_0 = approx_trans_prob(step_size, num_evals, theta_0);
  arma::mat trans_prob = approx_trans_prob(step_size, num_evals, theta_exp);
  arma::field<arma::vec> prior_p_labels = prior_Labels(n_AB, trans_prob_0, trans_prob);
  for(int i = 0; i < n_AB.n_elem; i++){
    arma::mat Prob_mat = forward_pass(theta_exp, X_AB(i, 0), trans_prob_0, trans_prob);
    arma::vec proposed_labels = backward_sim(Prob_mat, theta_exp, X_AB(i, 0), prob_propose);
    prob_current1 = prob_current(Labels(i, iter), Prob_mat, theta_exp, X_AB(i,0));
    prob_accept = posterior_Labels(proposed_labels, X_AB(i,0), theta_exp, prior_p_labels(i,0)) + prob_current1 -
      posterior_Labels(Labels(i, iter), X_AB(i,0), theta_exp, prior_p_labels(i,0)) - prob_propose;
    
    if(std::log(R::runif(0,1)) < prob_accept){
      accept_num = accept_num + 1;
      Labels(i, iter) = proposed_labels;
    }
  }
}

inline double calc_log_sum(arma::vec x){
  double max_val = x.max();
  
  double inner_sum = 0;
  for(int i = 0; i < x.n_elem; i++){
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
  l_prob = l_prob + alpha * R::dlnorm(delta, delta_proposal_mean, delta_proposal_sd, false);
  return std::log(l_prob);
}

inline void FFBS_ensemble_step(arma::field<arma::vec>& Labels,
                               int iter,
                               const arma::field<arma::vec>& X_AB,
                               const arma::vec& n_AB,
                               arma::vec& theta,
                               const double step_size,
                               const int num_evals,
                               double delta_proposal_mean,
                               double delta_proposal_sd,
                               const double alpha,
                               int M_proposal,
                               const double& delta_shape,
                               const double& delta_rate){
  arma::field<arma::vec> Labels_ensembles(n_AB.n_elem, M_proposal);
  arma::field<arma::vec> Labels_ensembles_ph(n_AB.n_elem, 1);
  arma::field<arma::vec> prior_labels_ph(n_AB.n_elem, 1);
  arma::vec theta_exp = arma::exp(theta);
  arma::vec theta_0 = theta_exp;
  theta_0(4) = 0;
  arma::mat trans_prob_0 = approx_trans_prob(step_size, num_evals, theta_0);
  arma::mat trans_prob = trans_prob_0;
  arma::vec delta_ensemble = arma::zeros(M_proposal);
  // Set initial positions of delta ensemble
  delta_ensemble(0) = theta_exp(4);
  //delta_ensemble(1) = theta_exp(4);
  for(int i = 1; i < M_proposal; i++){
    if(R::runif(0,1) < alpha){
      delta_ensemble(i) = R::rgamma(delta_shape, (1 / delta_rate));
    }else{
      delta_ensemble(i) = R::rlnorm(delta_proposal_mean, delta_proposal_sd);
    }
    
  }
  
  // Set first position of Labels to previous position
  for(int i = 0; i < n_AB.n_elem; i++){
    Labels_ensembles(i,0) = Labels(i, iter);
  }
  
  arma::vec theta_j = theta_exp;
  double prob_propose = 0;
  arma::vec q_L = arma::zeros(M_proposal); 
  arma::vec w_L = arma::zeros(M_proposal);
  arma::vec f_delta = arma::zeros(M_proposal);
  arma::vec f_delta_L = arma::zeros(M_proposal);
  
  // for delta 0
  trans_prob = approx_trans_prob(step_size, num_evals, theta_j);
  for(int i = 0; i < n_AB.n_elem; i++){
    arma::mat Prob_mat = forward_pass(theta_j, X_AB(i, 0), trans_prob_0, trans_prob);
    prob_propose = prob_propose + prob_current(Labels_ensembles(i, 0), Prob_mat,
                                               theta_j, X_AB(i,0));
  }
  for(int k = 0; k < M_proposal; k++){
    theta_j(4) = delta_ensemble(k);
    trans_prob = approx_trans_prob(step_size, num_evals, theta_j);
    prior_labels_ph = prior_Labels(n_AB, trans_prob_0, trans_prob);
    for(int i = 0; i < n_AB.n_elem; i++){
      Labels_ensembles_ph(i,0) = Labels_ensembles(i,0);
    }
    f_delta_L(k) = posterior_Labels_delta(Labels_ensembles_ph, X_AB,
            n_AB, theta_j, prior_labels_ph, delta_shape, delta_rate) - 
              log_prop_q(delta_ensemble(k), delta_proposal_mean, delta_proposal_sd,
                         delta_shape, delta_rate, alpha);
  }
  f_delta(0) = calc_log_sum(f_delta_L);
  q_L(0) = prob_propose;
  w_L(0) = f_delta(0) - q_L(0);
  
  
  // for the rest of the deltas
  for(int j = 1; j < M_proposal; j++){
    theta_j(4) = delta_ensemble(j);
    trans_prob = approx_trans_prob(step_size, num_evals, theta_j);
    for(int i = 0; i < n_AB.n_elem; i++){
      arma::mat Prob_mat = forward_pass(theta_j, X_AB(i, 0), trans_prob_0, trans_prob);
      Labels_ensembles(i,j) = backward_sim(Prob_mat, theta_j, X_AB(i, 0), prob_propose);
    }
    for(int i = 0; i < n_AB.n_elem; i++){
      Labels_ensembles_ph(i,0) = Labels_ensembles(i,j);
    }
    for(int k = 0; k < M_proposal; k++){
      theta_j(4) = delta_ensemble(k);
      trans_prob = approx_trans_prob(step_size, num_evals, theta_j);
      prior_labels_ph = prior_Labels(n_AB, trans_prob_0, trans_prob);
      
      f_delta_L(k) = posterior_Labels_delta(Labels_ensembles_ph, X_AB,
                n_AB, theta_j, prior_labels_ph, delta_shape, delta_rate) - 
                  log_prop_q(delta_ensemble(k), delta_proposal_mean, delta_proposal_sd,
                             delta_shape, delta_rate, alpha);
    }
    f_delta(j) = calc_log_sum(f_delta_L);
    
    q_L(j) = prob_propose;
    w_L(j) = f_delta(j) - q_L(j);
    prob_propose = 0;
  }
  
  arma::vec probs = arma::zeros(M_proposal);
  for(int i = 0; i < M_proposal; i++){
    probs(i) = exp(w_L(i) - calc_log_sum(w_L));
  }
  
  arma::vec draw = rmutlinomial(probs);
  
  // Update Labels
  for(int i = 0; i < draw.n_elem; i++){
    if(draw(i) == 1){
      Rcpp::Rcout << probs(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, iter) = Labels_ensembles(j, i);
      }
    }
  }
  
  // Update delta
  f_delta = arma::zeros(M_proposal);
  arma::vec w_delta = arma::zeros(M_proposal);
  theta_j = theta_exp;
  for(int i = 0; i < n_AB.n_elem; i++){
    Labels_ensembles_ph(i,0) = Labels(i, iter);
  }
  for(int j = 0; j < M_proposal; j++){
    theta_j(4) = delta_ensemble(j);
    w_delta(j) = posterior_Labels_delta(Labels_ensembles_ph, X_AB,
                          n_AB, theta_j, prior_labels_ph, delta_shape, delta_rate) - 
                            log_prop_q(delta_ensemble(j), delta_proposal_mean, delta_proposal_sd,
                                       delta_shape, delta_rate, alpha);
  }
  probs = arma::zeros(M_proposal);
  for(int i = 0; i < M_proposal; i++){
    probs(i) = exp(w_delta(i) - calc_log_sum(w_delta));
  }
  draw = rmutlinomial(probs);
  // Update Labels
  for(int i = 0; i < draw.n_elem; i++){
    if(draw(i) == 1){
      theta(4) = std::log(delta_ensemble(i));
    }
  }

}


// inline void FFBS_joint_step(arma::field<arma::vec>& Labels,
//                             int iter,
//                             const arma::field<arma::vec>& X_AB,
//                             const arma::vec& n_AB,
//                             arma::mat theta,
//                             const double step_size,
//                             const int num_evals,
//                             const double prior_p_labels,
//                             const double delta_shape_proposal){
//   double prob_propose = 0;
//   double prob_current1 = 0;
//   double prob_accept = 0;
//   arma::vec theta_current = theta.row(iter);
//   // propose new step
//   arma::vec theta_propose = theta_current;
//   theta_propose(4) = rinv_gauss(theta_current(4), delta_shape_proposal);
//   arma::vec theta_propose_exp = arma::exp(theta_propose);
//   arma::vec theta_current_exp = arma::exp(theta_current);
//   arma::vec theta_0 = theta_propose_exp;
//   theta_0(4) = 0;
//   arma::mat trans_prob_0 = approx_trans_prob(step_size, num_evals, theta_0);
//   arma::mat trans_prob = approx_trans_prob(step_size, num_evals, theta_current_exp);
//   for(int i = 0; i < n_AB.n_elem; i++){
//     arma::mat Prob_mat = forward_pass(theta_propose_exp, X_AB(i, 0), trans_prob_0, trans_prob);
//     arma::vec proposed_labels = backward_sim(Prob_mat, theta_propose_exp, X_AB(i, 0), prob_propose);
//     arma::mat Prob_mat_current = forward_pass(theta_current_exp, X_AB(i, 0), trans_prob_0, trans_prob);
//     prob_current1 = prob_current(Labels(i, iter), Prob_mat_current, theta_current_exp, X_AB(i,0));
//     prob_accept = posterior_Labels(proposed_labels, X_AB(i,0), theta_exp, prior_p_labels) + prob_current1 -
//       posterior_Labels(Labels(i, iter), X_AB(i,0), theta_exp, prior_p_labels) - prob_propose;
//     
//     if(std::log(R::runif(0,1)) < prob_accept){
//       Labels(i, iter) = proposed_labels;
//     }
//   }
// }

inline arma::field<arma::vec> FFBS(const arma::field<arma::vec>& X_AB,
                                   const arma::field<arma::vec>& init_position,
                                   const arma::vec& n_AB,
                                   arma::vec& theta,
                                   const double step_size,
                                   const int num_evals,
                                   int MCMC_iters){
  
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters);
  // Use initial starting position
  for(int j = 0; j < MCMC_iters; j++){
    for(int i = 0; i < n_AB.n_elem; i++){
      Labels(i, j) = init_position(i,0);
    }
  }
  int accept_num = 0;
  
  for(int i = 1; i < MCMC_iters; i++){
    //FFBS_step(Labels, i, X_AB, n_AB, theta, step_size, num_evals, prior_p_labels, accept_num);
    if((i + 1) < MCMC_iters){
      for(int j = 0; j < X_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
    if( (i % 50) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
  }
  Rcpp::Rcout << accept_num;
  return Labels;
}

}

#endif