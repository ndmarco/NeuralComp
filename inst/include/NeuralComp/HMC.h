#ifndef NeuralComp_HMC_H
#define NeuralComp_HMC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Posterior.h"
#include "Labels.h"

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
                          const arma::vec& eps_step,
                          double& step_size,
                          arma::mat& Mass_mat,
                          arma::vec& position,
                          arma::vec& momentum){
  arma::vec momentum_i = momentum + 0.5 * step_size * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        eps_step);
  position.subvec(0,3) = position.subvec(0,3) + step_size * arma::inv_sympd(Mass_mat) * momentum;
  momentum = momentum_i + 0.5 * step_size * 
    trans_calc_gradient(Labels, position, X_A, X_B, X_AB, n_A,
                        n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                        sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                        eps_step);
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
                     const double& I_A_shape, 
                     const double& I_A_rate,
                     const double& I_B_shape,
                     const double& I_B_rate,
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
                  n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
                  sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
                  eps_step, step_size, Mass_mat,
                  prop_position, prop_momentum);
  }
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
                     double& num_accept,
                     double& num_accept_delta){
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
  
  // Sample for delta
  double momentum_delta = R::rnorm(0,1);
  prop_position = theta;
  double prop_momentum_delta = momentum_delta;
  leapfrog_delta(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, delta_shape, delta_rate,
                 eps_step, step_size_delta, theta, momentum_delta, prop_position,
                 prop_momentum_delta, Leapfrog_steps);
  double accept_delta = lprob_accept_delta(prop_position, prop_momentum_delta, theta,
                                           momentum_delta, Labels, X_A, X_B, X_AB, n_A,
                                           n_B, n_AB, delta_shape, delta_rate);
  if(std::log(R::runif(0,1)) < accept_delta){
    num_accept_delta = 1;
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
             vec_accept(i), vec_accept_delta(i));
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
             vec_accept(i), vec_accept_delta(i));
    theta.row(i) = theta_ph.t();
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
    }
  }
  
  return arma::exp(theta);
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
  double prop_accept_10_delta = 0;
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters + Warm_block);
  arma::field<arma::vec> Labels_iter(n_AB.n_elem, 1);
  double llik = 0;
  // Use initial starting position
  //Rcpp::Rcout << "Made it";
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < MCMC_iters + Warm_block; j++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
    Labels_iter(i,0) = arma::zeros(n_AB(i));
  }
 
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
  
    llik = log_likelihood(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB);
    Rcpp::Rcout << llik;
    HMC_step(Labels_iter, theta_ph, X_A, X_B, X_AB, n_A, n_B, n_AB, I_A_shape, 
             I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
             eps_step, Mass_mat, step_size, step_size_delta, Leapfrog_steps,
             vec_accept(i), vec_accept_delta(i));
    theta.row(i) = theta_ph.t();
    FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, prior_p_labels);
    
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
      prop_accept_10_delta = arma::accu(vec_accept_delta.subvec(i-9, i))/ 10;
      if(prop_accept_10_delta  <= 0.1){
        step_size_delta = step_size_delta * 0.1;
      }else if(prop_accept_10_delta <= 0.3){
        step_size_delta = step_size_delta * 0.5;
      }else if(prop_accept_10_delta <= 0.6){
        step_size_delta = step_size_delta * 0.8;
      }else if(prop_accept_10_delta > 0.85){
        step_size_delta = step_size_delta * 1.1;
      }
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
             vec_accept(i), vec_accept_delta(i));
    theta.row(i) = theta_ph.t();
    FFBS_step(Labels, i, X_AB, n_AB, theta_ph, step_size_labels, num_evals, prior_p_labels);
    if((i+1) < Warm_block + MCMC_iters){
      theta.row(i+1) = theta.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(theta)),
                                         Rcpp::Named("labels", Labels));
  
  return params;
}

}


#endif