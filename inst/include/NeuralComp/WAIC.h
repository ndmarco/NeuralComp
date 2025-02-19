#ifndef WAIC_H
#define WAIC_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Priors.h"
#include "Labels.h"
#include <splines2Armadillo.h>

namespace NeuralComp {

// inline arma::mat approximate_L_given_theta(arma::vec theta,
//                                            arma::vec basis_coef_A,
//                                            arma::vec basis_coef_B,
//                                            arma::vec basis_func,
//                                            double max_time,
//                                            int n_eval){
//   arma::mat P_mat(2, 2, arma::fill::zeros);
//   arma::vec eval = arma::linspace(0, max_time, n_eval);
//   double step_size = eval(1) - eval(0);
//   
//   for(int i = 0; i < n_eval; i++){
//     P_mat(0,0) = P_mat(0,0) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)) +
//       dinv_gauss(eval(i), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2))) * step_size;
//     P_mat(1,1) = P_mat(1,1) + std::exp(pinv_gauss(eval(i) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)) +
//       dinv_gauss(eval(i), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2))) * step_size;
//   }
//   
//   eval = arma::linspace(theta(4), max_time + theta(4), n_eval);
//   for(int i = 0; i < n_eval; i++){
//     P_mat(0,1) = P_mat(0,1) + std::exp(pinv_gauss(eval(i), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)) +
//       dinv_gauss(eval(i) - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2))) * step_size;
//     P_mat(1,0) = P_mat(1,0) + std::exp(pinv_gauss(eval(i), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)) +
//       dinv_gauss(eval(i) - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2))) * step_size;
//   }
//   
//   P_mat(0,0) = (P_mat(0,0) + (1 - P_mat(0,1))) / 2;
//   P_mat(0,1) = 1 - P_mat(0,0);
//   P_mat(1,0) = (P_mat(1,0) + (1 - P_mat(1,1))) / 2;
//   P_mat(1,1) = 1 - P_mat(1,0);
//   
//   return P_mat;
// }
// 
// inline double joint_prob00(arma::vec theta,
//                          arma::vec basis_coef_A,
//                          arma::vec basis_coef_B,
//                          arma::vec basis_func,
//                          double x){
//   double output = std::exp(pinv_gauss(x - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)) +
//     dinv_gauss(x, (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)));
//   
//   return output;
// }
// 
// inline double joint_prob01(arma::vec theta,
//                            arma::vec basis_coef_A,
//                            arma::vec basis_coef_B,
//                            arma::vec basis_func,
//                            double x){
//   double output = std::exp(pinv_gauss(x, (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)) +
//                            dinv_gauss(x - theta(4), (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)));
//   
//   return output;
// }
// 
// inline double joint_prob10(arma::vec theta,
//                            arma::vec basis_coef_A,
//                            arma::vec basis_coef_B,
//                            arma::vec basis_func,
//                            double x){
//   double output = std::exp(pinv_gauss(x, (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)) +
//                         dinv_gauss(x - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)));
//   
//   return output;
// }
// 
// inline double joint_prob11(arma::vec theta,
//                            arma::vec basis_coef_A,
//                            arma::vec basis_coef_B,
//                            arma::vec basis_func,
//                            double x){
//   double output = std::exp(pinv_gauss(x - theta(4), (1 / (theta(0) * std::exp(arma::dot(basis_coef_A, basis_func)))), std::pow((1 / theta(2)), 2)) +
//                            dinv_gauss(x, (1 / (theta(1) * std::exp(arma::dot(basis_coef_B, basis_func)))), std::pow((1 / theta(3)), 2)));
//   
//   return output;
// }
// 
// inline arma::mat approximate_L_given_theta_simpson(arma::vec theta,
//                                                    arma::vec basis_coef_A,
//                                                    arma::vec basis_coef_B,
//                                                    arma::vec basis_func,
//                                                    double max_time,
//                                                    int n_eval){
//   arma::mat P_mat(2, 2, arma::fill::zeros);
//   arma::vec eval = arma::linspace(0, max_time, (3 * n_eval) + 1);
//   double step_size = eval(1) - eval(0);
//   
//   double ph = 0;
//   for(int i = 0; i < eval.n_elem; i++){
//     if(i == 0){
//       ph = ph + joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if(i == (eval.n_elem - 1)){
//       ph = ph + joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if((i % 3) != 0){
//       ph = ph + 3 * joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else{
//       ph = ph + 2 * joint_prob00(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }
//   }
//   P_mat(0,0) = (3 * step_size * ph) / 8;
//   
//   ph = 0;
//   for(int i = 0; i < eval.n_elem; i++){
//     if(i == 0){
//       ph = ph + joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if(i == (eval.n_elem - 1)){
//       ph = ph + joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if((i % 3) != 0){
//       ph = ph + 3 * joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else{
//       ph = ph + 2 * joint_prob01(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }
//   }
//   P_mat(0,1) = (3 * step_size * ph) / 8;
//   
//   ph = 0;
//   for(int i = 0; i < eval.n_elem; i++){
//     if(i == 0){
//       ph = ph + joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if(i == (eval.n_elem - 1)){
//       ph = ph + joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if((i % 3) != 0){
//       ph = ph + 3 * joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else{
//       ph = ph + 2 * joint_prob10(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }
//   }
//   P_mat(1,0) = (3 * step_size * ph) / 8;
//   
//   ph = 0;
//   for(int i = 0; i < eval.n_elem; i++){
//     if(i == 0){
//       ph = ph + joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if(i == (eval.n_elem - 1)){
//       ph = ph + joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else if((i % 3) != 0){
//       ph = ph + 3 * joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }else{
//       ph = ph + 2 * joint_prob11(theta, basis_coef_A, basis_coef_B, basis_func, eval(i));
//     }
//   }
//   P_mat(1,1) = (3 * step_size * ph) / 8;
//   if(((P_mat(0,0) + (1 - P_mat(0,1))) / 2) > 0){
//     P_mat(0,0) = (P_mat(0,0) + (1 - P_mat(0,1))) / 2;
//   }
//   P_mat(0,1) = 1 - P_mat(0,0);
//   if(((P_mat(1,0) + (1 - P_mat(1,1))) / 2) > 0){
//     P_mat(1,0) = (P_mat(1,0) + (1 - P_mat(1,1))) / 2;
//   }
//   P_mat(1,1) = 1 - P_mat(1,0);
//   
//   return P_mat;
// }


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
      for(int j = 0; j < X_A.n_elem; j++){
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
      for(int j = 0; j < X_B.n_elem; j++){
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

// inline arma::vec calc_loglikelihood_AB(const arma::vec X_AB,
//                                        const arma::mat theta,
//                                        const arma::mat basis_coef_A,
//                                        const arma::mat basis_coef_B,
//                                        const arma::mat basis_funct_AB,
//                                        const arma::field<arma::mat> Labels,
//                                        arma::field<arma::mat> P_mat,
//                                        const int obs_num,
//                                        const int spike_num,
//                                        const double burnin_prop,
//                                        const int P_mat_index,
//                                        arma::vec time_vec){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::vec llik = arma::zeros(n_MCMC - burnin_num);
//   arma::mat P_mat_lin_inter = arma::zeros(2,2);
//   
//   for(int i = burnin_num; i < n_MCMC; i++){
//     if(time_vec.n_elem > 1){
//       P_mat_lin_inter = P_mat(i - burnin_num, P_mat_index) + (((arma::accu(X_AB.subvec(0, spike_num-1)) - time_vec(P_mat_index)) / (time_vec(P_mat_index + 1) - time_vec(P_mat_index))) * (P_mat(i - burnin_num, P_mat_index + 1) - P_mat(i - burnin_num, P_mat_index)));
//     }else{
//       P_mat_lin_inter = P_mat(i - burnin_num, P_mat_index);
//     }
//     if(Labels(obs_num, 0)(i,spike_num) == 0){
//       // label is A
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
//           // Condition if spike has not switched (still in A)
//           llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                                          std::pow((1 / theta(i, 3)), 2)) +
//                                                            dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2))) - std::log(P_mat_lin_inter(0, 0)));
//         }else{
//           // Condition if spike has switched from B to A
//           llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2)) +
//             dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2))) - std::log(P_mat_lin_inter(1, 0)));
//         }
//       }else{
//         llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2)) +
//           dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2))) - std::log(P_mat_lin_inter(0, 0)));
//       }
//     }else{
//       // label is B
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
//           // Condition if spike has not switched (still in B)
//           llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2)) +
//             dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2))) - std::log(P_mat_lin_inter(1, 1)));
//         }else{
//           // Condition if spike has switched from A to B
//           llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2)) +
//             dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2))) - std::log(P_mat_lin_inter(0, 1)));
//         }
//       }else{
//         llik(i - burnin_num) = llik(i - burnin_num) + ((pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2)) +
//           dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2))) - std::log(P_mat_lin_inter(1, 1)));
//       }
//     }
//   }
//   
//   return llik;
// }

// inline double draw_tilted_prob(double mean1,
//                                double sigma1,
//                                double mean2,
//                                double sigma2,
//                                double delta1,
//                                double delta2){
//   double output = -1;
//   double proposed;
//   double accept_prob;
//   while(output < 0){
//     proposed = rinv_gauss((1 / mean1), std::pow((1 / sigma1), 2)) + delta1;
//     accept_prob = 1 - std::exp(pinv_gauss(proposed - delta2, (1 / mean2), std::pow((1 /sigma2), 2)));
//     if(R::runif(0,1) < accept_prob){
//       output = proposed;
//     }
//   }
//   return output;
// }
// 
// inline double draw_tilted_prob_lower(double mean1,
//                                      double sigma1,
//                                      double mean2,
//                                      double sigma2,
//                                      double delta1,
//                                      double delta2){
//   double output = -1;
//   double proposed;
//   double accept_prob;
//   while(output < 0){
//     proposed = rinv_gauss((1 / mean1), std::pow((1 / sigma1), 2)) + delta1;
//     accept_prob = std::exp(pinv_gauss(proposed - delta2, (1 / mean2), std::pow((1 /sigma2), 2)));
//     if(R::runif(0,1) < accept_prob){
//       output = proposed;
//     }
//   }
//   return output;
// }
// 
// 
// inline double calc_w_IS(double x,
//                         double y,
//                         double mean1,
//                         double sigma1,
//                         double mean2,
//                         double sigma2,
//                         double delta1,
//                         double delta2,
//                         const int n_MCMC_approx){
//   arma::vec x_samp = arma::zeros(n_MCMC_approx);
//   arma::vec est_probs = arma::zeros(n_MCMC_approx);
//   for(int i = 0; i < n_MCMC_approx; i++){
//     x_samp(i) = draw_tilted_prob_lower(mean2, sigma2, mean1, sigma1, delta2, delta1);
//     est_probs(i) = std::exp(dinv_gauss_trunc(y - delta1, (1 / mean1), std::pow((1 / sigma1), 2), x_samp(i) - delta1, INFINITY));
//   }
//   double est_prob = arma::mean(est_probs);
//   
//   double output = est_prob / std::exp(dinv_gauss_trunc(y-delta1, (1 / mean1), std::pow((1 / sigma1), 2), x-delta1, INFINITY));
//   return output;
// }
// 
// inline double weighted_mean(arma::vec x,
//                             arma::vec w){
//   double output = 0;
//   for(int i = 0; i < x.n_elem; i++){
//     output = output + w(i) * x(i);
//   }
//   output = output / x.n_elem;
//   return output;
// }
// 
// inline double weighted_sd(arma::vec x,
//                           arma::vec w,
//                           double mean){
//   double output = 0;
//   for(int i = 0; i < x.n_elem; i++){
//     output = output + ((w(i) * x(i) - mean) * (w(i) * x(i) - mean));
//   }
//   output = output / (x.n_elem - 1);
//   return std::sqrt(output);
// }
// 
// inline void calc_loglikelihood_AB_MCMC_approx4(const arma::vec X_AB,
//                                                const arma::mat theta,
//                                                const arma::mat basis_coef_A,
//                                                const arma::mat basis_coef_B,
//                                                const arma::mat basis_funct_AB,
//                                                const arma::field<arma::mat> Labels,
//                                                const int obs_num,
//                                                const int spike_num,
//                                                const double burnin_prop,
//                                                const int n_MCMC_approx,
//                                                const int n_samples_var,
//                                                arma::vec& llik,
//                                                arma::vec& llik_sd){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::vec ph_sd = arma::zeros(n_samples_var);
//   arma::vec denom = arma::zeros(n_MCMC_approx);
//   arma::mat denomi = arma::zeros(n_MCMC_approx, n_samples_var - 1);
//   double numerator = 0;
//   double isi_fast = 0;
//   for(int i = burnin_num; i < n_MCMC; i++){
//     if(Labels(obs_num, 0)(i,spike_num) == 0){
//       // label is A
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
//           // Condition if spike has not switched (still in A)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                   std::pow((1 / theta(i, 2)), 2.0));
//             denom(j) = std::exp(pinv_gauss(isi_fast - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                        std::pow((1 / theta(i, 3)), 2.0)));
//             
//           }
//           numerator = (pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                   std::pow((1 / theta(i, 3)), 2.0)) +
//                                     dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2.0)));
//           llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//           ph_sd(0) = llik(i - burnin_num);
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                     std::pow((1 / theta(i, 2)), 2.0));
//               denomi(j,k) = std::exp(pinv_gauss(isi_fast - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                      std::pow((1 / theta(i, 3)), 2.0)));
//             }
//             ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//           }
//           
//           llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//         }else{
//           // Condition if spike has switched from B to A
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                   std::pow((1 / theta(i, 2)), 2.0)) + theta(i, 4);
//             denom(j) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                             std::pow((1 / theta(i, 3)), 2.0)));
//           }
//           numerator = (pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                   std::pow((1 / theta(i, 3)), 2.0)) +
//                                     dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2.0)));
//           llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//           ph_sd(0) = llik(i - burnin_num);
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                     std::pow((1 / theta(i, 2)), 2.0)) + theta(i, 4);
//               denomi(j,k) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                      std::pow((1 / theta(i, 3)), 2.0)));
//             }
//             ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//           }
//           llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                 std::pow((1 / theta(i, 2)), 2.0));
//           denom(j) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                           std::pow((1 / theta(i, 3)), 2.0)));
//         }
//         numerator = (pinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                 std::pow((1 / theta(i, 3)), 2.0)) +
//                                   dinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2.0)));
//         llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//         ph_sd(0) = llik(i - burnin_num);
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                   std::pow((1 / theta(i, 2)), 2.0));
//             denomi(j,k) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                    std::pow((1 / theta(i, 3)), 2.0)));
//           }
//           ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//         }
//         llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//       }
//     }else{
//       // label is B
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
//           // Condition if spike has not switched (still in B)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                   std::pow((1 / theta(i, 3)), 2.0));
//             denom(j) = std::exp(pinv_gauss(isi_fast - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                             std::pow((1 / theta(i, 2)), 2.0)));
//           }
//           numerator = (pinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                   std::pow((1 / theta(i, 2)), 2.0)) +
//                                     dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2.0)));
//           llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//           ph_sd(0) = llik(i - burnin_num);
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                     std::pow((1 / theta(i, 3)), 2.0));
//               denomi(j,k) = std::exp(pinv_gauss(isi_fast - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                      std::pow((1 / theta(i, 2)), 2.0)));
//             }
//             ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//           }
//           llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//         }else{
//           // Condition if spike has switched from A to B
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                   std::pow((1 / theta(i, 3)), 2.0)) + theta(i, 4);
//             denom(j) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                                              std::pow((1 / theta(i, 2)), 2.0)));
//           }
//           numerator = (pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                         std::pow((1 / theta(i, 2)), 2.0)) +
//                                           dinv_gauss(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2.0)));
//           llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//           ph_sd(0) = llik(i - burnin_num);
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                     std::pow((1 / theta(i, 3)), 2.0)) + theta(i, 4);
//               denomi(j,k) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                      std::pow((1 / theta(i, 2)), 2.0)));
//             }
//             ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//           }
//           llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                 std::pow((1 / theta(i, 3)), 2.0));
//           denom(j) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                               std::pow((1 / theta(i, 2)), 2.0)));
//         }
//         numerator = (pinv_gauss(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                         std::pow((1 / theta(i, 2)), 2.0)) +
//                                           dinv_gauss(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2.0)));
//         llik(i - burnin_num) = numerator - std::log(arma::mean(denom));
//         ph_sd(0) = llik(i - burnin_num);
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_fast = rinv_gauss((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))),
//                                   std::pow((1 / theta(i, 3)), 2.0));
//             denomi(j,k) = std::exp(pinv_gauss(isi_fast, (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))),
//                                    std::pow((1 / theta(i, 2)), 2.0)));
//           }
//           ph_sd(k + 1) = numerator - std::log(arma::mean(denomi.col(k)));
//         }
//         llik_sd(i - burnin_num) = arma::stddev(ph_sd);
//       }
//     }
//   }
// }
// 
// // inline void calc_loglikelihood_AB_MCMC_approx2(const arma::vec X_AB,
// //                                                const arma::mat theta,
// //                                                const arma::mat basis_coef_A,
// //                                                const arma::mat basis_coef_B,
// //                                                const arma::mat basis_funct_AB,
// //                                                const arma::field<arma::mat> Labels,
// //                                                const int obs_num,
// //                                                const int spike_num,
// //                                                const double burnin_prop,
// //                                                const int n_MCMC_approx,
// //                                                arma::vec& llik,
// //                                                arma::vec& llik_sd){
// //   int n_MCMC = theta.n_rows;
// //   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
// //   arma::vec llik_samples = arma::zeros(n_MCMC_approx + 1);
// //   double isi_slow = 0;
// //   
// //   for(int i = burnin_num; i < n_MCMC; i++){
// //     if(Labels(obs_num, 0)(i,spike_num) == 0){
// //       // label is A
// //       if(spike_num != 0){
// //         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
// //           // Condition if spike has not switched (still in A)
// //           for(int j = 0; j < n_MCMC_approx; j++){
// //             isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                         theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                         theta(i,2), theta(i,4), 0);
// //             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
// //           
// //           }
// //           isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
// //           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
// //           llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //           if(n_MCMC_approx > 1){
// //             llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //           }
// //           
// //         }else{
// //           // Condition if spike has switched from B to A
// //           for(int j = 0; j < n_MCMC_approx; j++){
// //             isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                         theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                         theta(i,2), 0, theta(i,4));
// //             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
// //           }
// //           isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
// //           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
// //           llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //           if(n_MCMC_approx > 1){
// //             llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //           }
// //         }
// //       }else{
// //         for(int j = 0; j < n_MCMC_approx; j++){
// //           isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                       theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                       theta(i,2), 0, 0);
// //           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
// //         }
// //         isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
// //         llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
// //         llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //         if(n_MCMC_approx > 1){
// //           llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //         }
// //       }
// //     }else{
// //       // label is B
// //       if(spike_num != 0){
// //         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
// //           // Condition if spike has not switched (still in B)
// //           for(int j = 0; j < n_MCMC_approx; j++){
// //             isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                         theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                         theta(i,3), theta(i,4), 0);
// //             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
// //           }
// //           isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
// //           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
// //           llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //           if(n_MCMC_approx > 1){
// //             llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //           }
// //         }else{
// //           // Condition if spike has switched from A to B
// //           for(int j = 0; j < n_MCMC_approx; j++){
// //             isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                         theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                         theta(i,3), 0, theta(i,4));
// //             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
// //           }
// //           isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
// //           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
// //           llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //           if(n_MCMC_approx > 1){
// //             llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //           }
// //         }
// //       }else{
// //         for(int j = 0; j < n_MCMC_approx; j++){
// //           isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
// //                                       theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
// //                                       theta(i,3), 0, 0);
// //           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
// //         }
// //         isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
// //         llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
// //         llik(i - burnin_num) = std::log(arma::mean(llik_samples));
// //         if(n_MCMC_approx > 1){
// //           llik_sd(i - burnin_num) = arma::stddev(llik_samples) * (1 / std::exp(llik(i - burnin_num)));
// //         }
// //       }
// //     }
// //   }
// // }
// 
// inline void calc_loglikelihood_AB_MCMC_approx2(const arma::vec X_AB,
//                                                const arma::mat theta,
//                                                const arma::mat basis_coef_A,
//                                                const arma::mat basis_coef_B,
//                                                const arma::mat basis_funct_AB,
//                                                const arma::field<arma::mat> Labels,
//                                                const int obs_num,
//                                                const int spike_num,
//                                                const double burnin_prop,
//                                                const int n_MCMC_approx,
//                                                const int n_MCMC_approx2,
//                                                arma::vec& llik,
//                                                arma::vec& llik_sd){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::vec llik_samples = arma::zeros(n_MCMC_approx);
//   arma::vec weights = arma::ones(n_MCMC_approx);
//   double isi_slow = 0;
//   for(int i = burnin_num; i < n_MCMC; i++){
//     if(Labels(obs_num, 0)(i,spike_num) == 0){
//       // label is A
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
//           // Condition if spike has not switched (still in A)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//         }else{
//           // Condition if spike has switched from B to A
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//           weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), 0, 0, n_MCMC_approx2);
//         }
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//       }
//     }else{
//       // label is B
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
//           // Condition if spike has not switched (still in B)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//         }else{
//           // Condition if spike has switched from A to B
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), 0, 0, n_MCMC_approx2);
//         }
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         llik_sd(i - burnin_num) = weighted_sd(llik_samples, weights, std::exp(llik(i - burnin_num))) * (1 / std::exp(llik(i - burnin_num)));
//       }
//     }
//   }
// }
// 
// inline double calc_unbiased_sd(arma::vec x,
//                                arma::vec w,
//                                arma::mat xi,
//                                arma::mat wi){
//   arma::vec estimates = arma::zeros(wi.n_cols + 1);
//   for(int i = 0; i < x.n_elem; i++){
//     estimates(0) = estimates(0) + (w(i) * x(i));
//   }
//   estimates(0) = std::log(estimates(0) / x.n_elem);
//   for(int j = 0; j < wi.n_cols; j++){
//     for(int i = 0; i < x.n_elem; i++){
//       estimates(j + 1) = estimates(j + 1) + (wi(i,j) * xi(i, j));
//     }
//     estimates(j+1) = std::log(estimates(j+1) / x.n_elem);
//   }
//   double output = arma::stddev(estimates);
//   return output;
// }
// 
// inline void calc_loglikelihood_AB_MCMC_approx(const arma::vec X_AB,
//                                               const arma::mat theta,
//                                               const arma::mat basis_coef_A,
//                                               const arma::mat basis_coef_B,
//                                               const arma::mat basis_funct_AB,
//                                               const arma::field<arma::mat> Labels,
//                                               const int obs_num,
//                                               const int spike_num,
//                                               const double burnin_prop,
//                                               const int n_MCMC_approx,
//                                               const int n_MCMC_approx2,
//                                               const int n_samples_var,
//                                               arma::vec& llik,
//                                               arma::vec& llik_sd){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::vec llik_samples = arma::zeros(n_MCMC_approx);
//   arma::mat llik_samplesi = arma::zeros(n_MCMC_approx, n_samples_var - 1);
//   arma::vec weights = arma::ones(n_MCMC_approx);
//   arma::mat weightsi = arma::ones(n_MCMC_approx, n_samples_var - 1);
//   bool weights_zero = false;
//   double isi_slow = 0;
//   for(int i = burnin_num; i < n_MCMC; i++){
//     if(Labels(obs_num, 0)(i,spike_num) == 0){
//       // label is A
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
//           // Condition if spike has not switched (still in A)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           if(std::exp(llik(i - burnin_num)) == 0){
//             weights_zero = true;
//           }
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//               weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                       theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                       theta(i,2), theta(i,4), 0, n_MCMC_approx2);
//             }
//           }
//           if(weights_zero == true){
//             for(int k = 0; k < (n_samples_var - 1); k++){
//               if(weights_zero == true){
//                 llik_samples = llik_samplesi.col(k);
//                 weights = weightsi.col(k);
//                 llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//                 if(std::exp(llik(i - burnin_num)) == 0){
//                   weights_zero = true;
//                 }
//               }
//             }
//             llik_sd(i - burnin_num)  = 0;
//           }else{
//             llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//           }
//           weights_zero = false;
//         }else{
//           // Condition if spike has switched from B to A
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           if(std::exp(llik(i - burnin_num)) == 0){
//             weights_zero = true;
//           }
//           
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//               weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                       theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                       theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//             }
//           }
//           if(weights_zero == true){
//             for(int k = 0; k < (n_samples_var - 1); k++){
//               if(weights_zero == true){
//                 llik_samples = llik_samplesi.col(k);
//                 weights = weightsi.col(k);
//                 llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//                 if(std::exp(llik(i - burnin_num)) == 0){
//                   weights_zero = true;
//                 }
//               }
//             }
//             llik_sd(i - burnin_num)  = 0;
//           }else{
//             llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//           }
//           weights_zero = false;
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//           weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), 0, 0, n_MCMC_approx2);
//         }
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         if(std::exp(llik(i - burnin_num)) == 0){
//           weights_zero = true;
//         }
//         
//         //calculate sd
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//             llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), 0, 0, n_MCMC_approx2);
//           }
//         }
//         if(weights_zero == true){
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             if(weights_zero == true){
//               llik_samples = llik_samplesi.col(k);
//               weights = weightsi.col(k);
//               llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//               if(std::exp(llik(i - burnin_num)) == 0){
//                 weights_zero = true;
//               }
//             }
//           }
//           llik_sd(i - burnin_num)  = 0;
//         }else{
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//         }
//         weights_zero = false;
//       }
//     }else{
//       // label is B
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
//           // Condition if spike has not switched (still in B)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           if(std::exp(llik(i - burnin_num)) == 0){
//             weights_zero = true;
//           }
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//               weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                       theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                       theta(i,3), theta(i,4), 0, n_MCMC_approx2);
//             }
//           }
//           if(weights_zero == true){
//             for(int k = 0; k < (n_samples_var - 1); k++){
//               if(weights_zero == true){
//                 llik_samples = llik_samplesi.col(k);
//                 weights = weightsi.col(k);
//                 llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//                 if(std::exp(llik(i - burnin_num)) == 0){
//                   weights_zero = true;
//                 }
//               }
//             }
//             llik_sd(i - burnin_num)  = 0;
//           }else{
//             llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//           }
//           weights_zero = false;
//         }else{
//           // Condition if spike has switched from A to B
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//             weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           if(std::exp(llik(i - burnin_num)) == 0){
//             weights_zero = true;
//           }
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//               weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                       theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                       theta(i,3), 0, theta(i,4), n_MCMC_approx2);
//             }
//           }
//           if(weights_zero == true){
//             for(int k = 0; k < (n_samples_var - 1); k++){
//               if(weights_zero == true){
//                 llik_samples = llik_samplesi.col(k);
//                 weights = weightsi.col(k);
//                 llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//                 if(std::exp(llik(i - burnin_num)) == 0){
//                   weights_zero = true;
//                 }
//               }
//             }
//             llik_sd(i - burnin_num)  = 0;
//           }else{
//             llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//           }
//           weights_zero = false;
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           weights(j) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), 0, 0, n_MCMC_approx2);
//         }
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         if(std::exp(llik(i - burnin_num)) == 0){
//           weights_zero = true;
//         }
//         //calculate sd
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//             llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//             weightsi(j,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), 0, 0, n_MCMC_approx2);
//           }
//         }
//         if(weights_zero == true){
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             if(weights_zero == true){
//               llik_samples = llik_samplesi.col(k);
//               weights = weightsi.col(k);
//               llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//               if(std::exp(llik(i - burnin_num)) == 0){
//                 weights_zero = true;
//               }
//             }
//           }
//           llik_sd(i - burnin_num)  = 0;
//         }else{
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//         }
//         weights_zero = false;
//       }
//     }
//   }
// }
// 
// inline void calc_loglikelihood_AB_MCMC_approx3(const arma::vec X_AB,
//                                                const arma::mat theta,
//                                                const arma::mat basis_coef_A,
//                                                const arma::mat basis_coef_B,
//                                                const arma::mat basis_funct_AB,
//                                                const arma::field<arma::mat> Labels,
//                                                const int obs_num,
//                                                const int spike_num,
//                                                const double burnin_prop,
//                                                const int n_MCMC_approx,
//                                                const int n_MCMC_approx2,
//                                                const int n_samples_var,
//                                                arma::vec& llik,
//                                                arma::vec& llik_sd){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::vec llik_samples = arma::zeros(n_MCMC_approx + 1);
//   arma::mat llik_samplesi = arma::zeros(n_MCMC_approx + 1, n_samples_var - 1);
//   double isi_slow = 0;
//   arma::vec weights = arma::ones(n_MCMC_approx + 1);
//   arma::mat weightsi = arma::ones(n_MCMC_approx + 1, n_samples_var - 1);
//   for(int i = burnin_num; i < n_MCMC; i++){
//     if(Labels(obs_num, 0)(i,spike_num) == 0){
//       // label is A
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i,spike_num - 1) == 0){
//           // Condition if spike has not switched (still in A)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), theta(i,4), 0);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             
//           }
//           isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//           weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), theta(i,4), 0, n_MCMC_approx2);
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                           theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                           theta(i,2), theta(i,4), 0);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             }
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//             weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//           
//         }else{
//           // Condition if spike has switched from B to A
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), 0, theta(i,4));
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//           }
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//           weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                           theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                           theta(i,2), 0, theta(i,4));
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//             }
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//             llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//             weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                       theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                       theta(i,2), 0, 0);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//         }
//         isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//         llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//         weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                 theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                 theta(i,2), 0, 0, n_MCMC_approx2);
//         
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         //calculate sd
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), 0, 0);
//             llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow));
//           }
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), X_AB(spike_num), INFINITY);
//           llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i, 4), (1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), 0, isi_slow - theta(i,4)));
//           weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                    theta(i,3), (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                    theta(i,2), 0, theta(i,4), n_MCMC_approx2);
//         }
//         llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//       }
//     }else{
//       // label is B
//       if(spike_num != 0){
//         if(Labels(obs_num, 0)(i, spike_num-1) == 1){
//           // Condition if spike has not switched (still in B)
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), theta(i,4), 0);
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           }
//           isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), theta(i,4), 0, n_MCMC_approx2);
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                           theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                           theta(i,3), theta(i,4), 0);
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//             }
//             isi_slow = theta(i,4) + rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num) - theta(i, 4), INFINITY);
//             llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//             weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), theta(i,4), 0, n_MCMC_approx2);
//           }
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//         }else{
//           // Condition if spike has switched from A to B
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), 0, theta(i,4));
//             llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//           }
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//           llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//           weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), 0, theta(i,4), n_MCMC_approx2);
//           llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//           //calculate sd
//           for(int k = 0; k < (n_samples_var - 1); k++){
//             for(int j = 0; j < n_MCMC_approx; j++){
//               isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                           theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                           theta(i,3), 0, theta(i,4));
//               llik_samplesi(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//             }
//             isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//             llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num) - theta(i,4), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow - theta(i,4)));
//             weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                     theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                     theta(i,3), 0, theta(i,4), n_MCMC_approx2);
//           }
//           llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//         }
//       }else{
//         for(int j = 0; j < n_MCMC_approx; j++){
//           isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                       theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                       theta(i,3), 0, 0);
//           llik_samples(j) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//         }
//         isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//         llik_samples(n_MCMC_approx) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//         weights(n_MCMC_approx) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                 theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                 theta(i,3), 0, 0, n_MCMC_approx2);
//         llik(i - burnin_num) = std::log(weighted_mean(llik_samples, weights));
//         //calculate sd
//         for(int k = 0; k < (n_samples_var - 1); k++){
//           for(int j = 0; j < n_MCMC_approx; j++){
//             isi_slow = draw_tilted_prob((theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                                         theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                                         theta(i,3), 0, 0);
//             llik_samples(j,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           }
//           isi_slow = rinv_gauss_trunc((1 / (theta(i, 0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i))))), std::pow((1 / theta(i, 2)), 2), X_AB(spike_num), INFINITY);
//           llik_samplesi(n_MCMC_approx,k) = std::exp(dinv_gauss_trunc(X_AB(spike_num), (1 / (theta(i, 1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i))))), std::pow((1 / theta(i, 3)), 2), 0, isi_slow));
//           weightsi(n_MCMC_approx,k) = calc_w_IS(X_AB(spike_num), isi_slow, (theta(i,0) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_A.row(i)))),
//                   theta(i,2), (theta(i,1) * std::exp(arma::dot(basis_funct_AB.row(spike_num), basis_coef_B.row(i)))),
//                   theta(i,3), 0, 0, n_MCMC_approx2);
//         }
//         llik_sd(i - burnin_num) = calc_unbiased_sd(llik_samples, weights, llik_samplesi, weightsi);
//       }
//     }
//   }
// }
// 
// inline Rcpp::List calc_WAIC_competition(const arma::field<arma::vec> X_A,
//                                         const arma::field<arma::vec> X_B,
//                                         const arma::field<arma::vec> X_AB,
//                                         const arma::vec n_A,
//                                         const arma::vec n_B,
//                                         const arma::vec n_AB,
//                                         const arma::mat theta,
//                                         const arma::mat basis_coef_A,
//                                         const arma::mat basis_coef_B,
//                                         const arma::field<arma::mat> basis_funct_A,
//                                         const arma::field<arma::mat> basis_funct_B,
//                                         const arma::field<arma::mat> basis_funct_AB,
//                                         const arma::field<arma::mat> Labels,
//                                         const double burnin_prop,
//                                         const double max_time,
//                                         const double max_spike_time,
//                                         const int n_spikes_eval,
//                                         const int n_eval,
//                                         const int basis_degree,
//                                         const arma::vec boundary_knots,
//                                         const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   
//   
//   arma::field<arma::mat> P_mat0(n_MCMC - burnin_num, 1);
//   arma::mat theta_0 = theta;
//   theta_0.col(4) = arma::zeros(theta_0.n_rows);
//   
//   for(int i = burnin_num; i < n_MCMC; i++){
//     P_mat0(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta_0.row(i).t(), basis_coef_A.row(i).t(),
//            basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
//   }
//   
//   arma::field<arma::mat> P_mat(n_MCMC - burnin_num, n_spikes_eval);
//   arma::vec time_vec;
//   if(max_spike_time != 0){
//     time_vec = arma::linspace(0, max_spike_time, n_spikes_eval);
//     splines2::BSpline bspline;
//     bspline = splines2::BSpline(time_vec, internal_knots, basis_degree,
//                                 boundary_knots);
//     // Get Basis matrix
//     arma::mat bspline_mat{bspline.basis(false)};
//     for(int j = 0; j < n_spikes_eval; j++){
//       for(int i = burnin_num; i < n_MCMC; i++){
//         P_mat(i - burnin_num, j) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
//                basis_coef_B.row(i).t(), bspline_mat.row(j).t(), max_time, n_eval);
//       }
//       Rcpp::Rcout << "Approximated transition probability for spike " << j + 1 << " out of " << n_spikes_eval << "\n";
//     }
//   }else{
//     for(int i = burnin_num; i < n_MCMC; i++){
//       P_mat(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
//              basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
//     }
//   }
//   
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//                basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//                basis_funct_B(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for AB
//   int index;
//   arma::vec time_vecph;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       if(max_spike_time != 0){
//         if(j == 0){
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0, time_vecph);
//         }else{
//           index = arma::index_min(arma::square(time_vec - arma::accu(X_AB(i,0).subvec(0,j-1))));
//           if((time_vec(index) - arma::accu(X_AB(i,0).subvec(0,j-1))) >= 0){
//             index = index - 1;
//           }
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, index, time_vec);
//         }
//       }else{
//         if(j == 0){
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0, time_vec);
//         }
//         else{
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, 0, time_vec);
//         }
//       }
//       
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + arma::var(llik_A(i,0).col(j));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + arma::var(llik_B(i,0).col(j));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
//     }
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                          Rcpp::Named("lppd", lppd),
//                                          Rcpp::Named("Effective_pars", pwaic),
//                                          Rcpp::Named("llik_A", llik_A),
//                                          Rcpp::Named("llik_B", llik_B),
//                                          Rcpp::Named("llik_AB", llik_AB));
//   return output;
// }
// 
// 
// inline Rcpp::List calc_WAIC_competition_approx(const arma::field<arma::vec> X_A,
//                                                const arma::field<arma::vec> X_B,
//                                                const arma::field<arma::vec> X_AB,
//                                                const arma::vec n_A,
//                                                const arma::vec n_B,
//                                                const arma::vec n_AB,
//                                                const arma::mat theta,
//                                                const arma::mat basis_coef_A,
//                                                const arma::mat basis_coef_B,
//                                                const arma::field<arma::mat> basis_funct_A,
//                                                const arma::field<arma::mat> basis_funct_B,
//                                                const arma::field<arma::mat> basis_funct_AB,
//                                                const arma::field<arma::mat> Labels,
//                                                const double burnin_prop,
//                                                const int n_MCMC_approx,
//                                                const int n_MCMC_approx2,
//                                                const int n_samples_var,
//                                                const int basis_degree,
//                                                const arma::vec boundary_knots,
//                                                const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   arma::field<arma::mat> llik_AB_sd(n_AB.n_elem, 1);
// 
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//     llik_AB_sd(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     arma::vec llik_ph = arma::zeros(n_MCMC - burnin_num);
//     arma::vec llik_sd_ph = arma::zeros(n_MCMC - burnin_num);
//     for(int j = 0; j < n_AB(i); j++){
//       calc_loglikelihood_AB_MCMC_approx(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
//                                         basis_funct_AB(i,0), Labels, i, j, burnin_prop,
//                                         n_MCMC_approx, n_MCMC_approx2, n_samples_var, 
//                                         llik_ph, llik_sd_ph);
//     
//       llik_AB(i,0).col(j) = llik_ph;
//       llik_AB_sd(i,0).col(j) = llik_sd_ph;
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   double excess_MCMC_var = 0;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + (arma::var(llik_AB(i,0).col(j)) - arma::mean(arma::square(llik_AB_sd(i,0).col(j))));
//       excess_MCMC_var = excess_MCMC_var + arma::mean(arma::square(llik_AB_sd(i,0).col(j)));
//     }
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "Excess variation due to imputation = " << excess_MCMC_var << "\n";
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }
// 
// inline Rcpp::List calc_WAIC_competition_approx_alt(const arma::field<arma::vec> X_A,
//                                                    const arma::field<arma::vec> X_B,
//                                                    const arma::field<arma::vec> X_AB,
//                                                    const arma::vec n_A,
//                                                    const arma::vec n_B,
//                                                    const arma::vec n_AB,
//                                                    const arma::mat theta,
//                                                    const arma::mat basis_coef_A,
//                                                    const arma::mat basis_coef_B,
//                                                    const arma::field<arma::mat> basis_funct_A,
//                                                    const arma::field<arma::mat> basis_funct_B,
//                                                    const arma::field<arma::mat> basis_funct_AB,
//                                                    const arma::field<arma::mat> Labels,
//                                                    const double burnin_prop,
//                                                    const int n_MCMC_approx,
//                                                    const int n_MCMC_approx2,
//                                                    const int basis_degree,
//                                                    const arma::vec boundary_knots,
//                                                    const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   arma::field<arma::mat> llik_AB_sd(n_AB.n_elem, 1);
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//     llik_AB_sd(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     arma::vec llik_ph = arma::zeros(n_MCMC - burnin_num);
//     arma::vec llik_sd_ph = arma::zeros(n_MCMC - burnin_num);
//     for(int j = 0; j < n_AB(i); j++){
//       calc_loglikelihood_AB_MCMC_approx2(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
//                                          basis_funct_AB(i,0), Labels, i, j, burnin_prop,
//                                          n_MCMC_approx, n_MCMC_approx2, llik_ph, llik_sd_ph);
//       
//       llik_AB(i,0).col(j) = llik_ph;
//       llik_AB_sd(i,0).col(j) = llik_sd_ph;
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   double excess_MCMC_var = 0;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + (arma::var(llik_AB(i,0).col(j)) - arma::mean(arma::square(llik_AB_sd(i,0).col(j)) / (n_MCMC_approx)));
//       excess_MCMC_var = excess_MCMC_var + arma::mean(arma::square(llik_AB_sd(i,0).col(j)) / (n_MCMC_approx));
//     }
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "Excess variation due to imputation = " << excess_MCMC_var << "\n";
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }
// 
// inline Rcpp::List calc_WAIC_competition_approx_alt_IS(const arma::field<arma::vec> X_A,
//                                                       const arma::field<arma::vec> X_B,
//                                                       const arma::field<arma::vec> X_AB,
//                                                       const arma::vec n_A,
//                                                       const arma::vec n_B,
//                                                       const arma::vec n_AB,
//                                                       const arma::mat theta,
//                                                       const arma::mat basis_coef_A,
//                                                       const arma::mat basis_coef_B,
//                                                       const arma::field<arma::mat> basis_funct_A,
//                                                       const arma::field<arma::mat> basis_funct_B,
//                                                       const arma::field<arma::mat> basis_funct_AB,
//                                                       const arma::field<arma::mat> Labels,
//                                                       const double burnin_prop,
//                                                       const int n_MCMC_approx,
//                                                       const int n_MCMC_approx2,
//                                                       const int n_samples_var,
//                                                       const int basis_degree,
//                                                       const arma::vec boundary_knots,
//                                                       const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   arma::field<arma::mat> llik_AB_sd(n_AB.n_elem, 1);
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//     llik_AB_sd(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     arma::vec llik_ph = arma::zeros(n_MCMC - burnin_num);
//     arma::vec llik_sd_ph = arma::zeros(n_MCMC - burnin_num);
//     for(int j = 0; j < n_AB(i); j++){
//       calc_loglikelihood_AB_MCMC_approx3(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
//                                          basis_funct_AB(i,0), Labels, i, j, burnin_prop,
//                                          n_MCMC_approx, n_MCMC_approx2, n_samples_var, llik_ph, llik_sd_ph);
//       
//       llik_AB(i,0).col(j) = llik_ph;
//       llik_AB_sd(i,0).col(j) = llik_sd_ph;
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   double excess_MCMC_var = 0;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + (arma::var(llik_AB(i,0).col(j)) - arma::mean(arma::square(llik_AB_sd(i,0).col(j))));
//       excess_MCMC_var = excess_MCMC_var + arma::mean(arma::square(llik_AB_sd(i,0).col(j)));
//     }
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "Excess variation due to imputation = " << excess_MCMC_var << "\n";
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }
// 
// inline Rcpp::List calc_WAIC_competition_approx_direct(const arma::field<arma::vec> X_A,
//                                                       const arma::field<arma::vec> X_B,
//                                                       const arma::field<arma::vec> X_AB,
//                                                       const arma::vec n_A,
//                                                       const arma::vec n_B,
//                                                       const arma::vec n_AB,
//                                                       const arma::mat theta,
//                                                       const arma::mat basis_coef_A,
//                                                       const arma::mat basis_coef_B,
//                                                       const arma::field<arma::mat> basis_funct_A,
//                                                       const arma::field<arma::mat> basis_funct_B,
//                                                       const arma::field<arma::mat> basis_funct_AB,
//                                                       const arma::field<arma::mat> Labels,
//                                                       const double burnin_prop,
//                                                       const int n_MCMC_approx,
//                                                       const int n_samples_var,
//                                                       const int basis_degree,
//                                                       const arma::vec boundary_knots,
//                                                       const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   arma::field<arma::mat> llik_AB_sd(n_AB.n_elem, 1);
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//     llik_AB_sd(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     arma::vec llik_ph = arma::zeros(n_MCMC - burnin_num);
//     arma::vec llik_sd_ph = arma::zeros(n_MCMC - burnin_num);
//     for(int j = 0; j < n_AB(i); j++){
//       calc_loglikelihood_AB_MCMC_approx4(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
//                                          basis_funct_AB(i,0), Labels, i, j, burnin_prop,
//                                          n_MCMC_approx, n_samples_var, llik_ph, llik_sd_ph);
//       
//       llik_AB(i,0).col(j) = llik_ph;
//       llik_AB_sd(i,0).col(j) = llik_sd_ph;
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   double excess_MCMC_var = 0;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + (arma::var(llik_AB(i,0).col(j)) - arma::mean(arma::square(llik_AB_sd(i,0).col(j))));
//       excess_MCMC_var = excess_MCMC_var + arma::mean(arma::square(llik_AB_sd(i,0).col(j)));
//     }
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "Excess variation due to imputation = " << excess_MCMC_var << "\n";
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }
// 
// inline double calc_WAIC_competition_observation(const arma::field<arma::vec> X_A,
//                                                 const arma::field<arma::vec> X_B,
//                                                 const arma::field<arma::vec> X_AB,
//                                                 const arma::vec n_A,
//                                                 const arma::vec n_B,
//                                                 const arma::vec n_AB,
//                                                 const arma::mat theta,
//                                                 const arma::mat basis_coef_A,
//                                                 const arma::mat basis_coef_B,
//                                                 const arma::field<arma::mat> basis_funct_A,
//                                                 const arma::field<arma::mat> basis_funct_B,
//                                                 const arma::field<arma::mat> basis_funct_AB,
//                                                 const arma::field<arma::mat> Labels,
//                                                 const double burnin_prop,
//                                                 const double max_time,
//                                                 const double max_spike_time,
//                                                 const int n_spikes_eval,
//                                                 const int n_eval,
//                                                 const int basis_degree,
//                                                 const arma::vec boundary_knots,
//                                                 const arma::vec internal_knots){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   
//   
//   arma::field<arma::mat> P_mat0(n_MCMC - burnin_num, 1);
//   arma::mat theta_0 = theta;
//   theta_0.col(4) = arma::zeros(theta_0.n_rows);
//   
//   for(int i = burnin_num; i < n_MCMC; i++){
//     P_mat0(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta_0.row(i).t(), basis_coef_A.row(i).t(),
//            basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
//   }
//   
//   arma::field<arma::mat> P_mat(n_MCMC - burnin_num, n_spikes_eval);
//   arma::vec time_vec;
//   if(max_spike_time != 0){
//     time_vec = arma::linspace(0, max_spike_time, n_spikes_eval);
//     splines2::BSpline bspline;
//     bspline = splines2::BSpline(time_vec, internal_knots, basis_degree,
//                                 boundary_knots);
//     // Get Basis matrix
//     arma::mat bspline_mat{bspline.basis(false)};
//     for(int j = 0; j < n_spikes_eval; j++){
//       for(int i = burnin_num; i < n_MCMC; i++){
//         P_mat(i - burnin_num, j) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
//               basis_coef_B.row(i).t(), bspline_mat.row(j).t(), max_time, n_eval);
//       }
//       Rcpp::Rcout << "Approximated transition probability for spike " << j + 1 << " out of " << n_spikes_eval << "\n";
//     }
//   }else{
//     for(int i = burnin_num; i < n_MCMC; i++){
//       P_mat(i - burnin_num, 0) = approximate_L_given_theta_simpson(theta.row(i).t(), basis_coef_A.row(i).t(),
//             basis_coef_B.row(i).t(), basis_funct_AB(0,0).row(0).t(), max_time, n_eval);
//     }
//   }
//   
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for AB
//   int index;
//   arma::vec time_vecph;
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       if(max_spike_time != 0){
//         if(j == 0){
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0, time_vecph);
//         }else{
//           index = arma::index_min(arma::square(time_vec - arma::accu(X_AB(i,0).subvec(0,j-1))));
//           if((time_vec(index) - arma::accu(X_AB(i,0).subvec(0,j-1))) >= 0){
//             index = index - 1;
//           }
//           Rcpp::Rcout << " " << index;
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, index, time_vec);
//         }
//       }else{
//         if(j == 0){
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat0, i, j, burnin_prop, 0, time_vec);
//         }
//         else{
//           llik_AB(i,0).col(j) = calc_loglikelihood_AB(X_AB(i,0), theta, basis_coef_A, basis_coef_B, 
//                   basis_funct_AB(i,0), Labels, P_mat, i, j, burnin_prop, 0, time_vec);
//         }
//       }
//       
//     }
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   arma::mat llik_A_obs = arma::zeros(n_MCMC - burnin_num, n_A.n_elem);
//   arma::mat llik_B_obs = arma::zeros(n_MCMC - burnin_num, n_B.n_elem);
//   arma::mat llik_AB_obs = arma::zeros(n_MCMC - burnin_num, n_AB.n_elem);
//   
//   for(int i = 0; i < n_MCMC - burnin_num; i++){
//     for(int j = 0; j < n_A.n_elem; j++){
//       llik_A_obs(i, j) = arma::accu(llik_A(j,0).row(i));
//     }
//     for(int j = 0; j < n_B.n_elem; j++){
//       llik_B_obs(i, j) = arma::accu(llik_B(j,0).row(i));
//     }
//     for(int j = 0; j < n_AB.n_elem; j++){
//       llik_AB_obs(i, j) = arma::accu(llik_AB(j,0).row(i));
//     }
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     lppd = lppd + std::log(arma::mean(arma::exp(llik_A_obs.col(i))));
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     lppd = lppd + std::log(arma::mean(arma::exp(llik_B_obs.col(i))));
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     lppd = lppd + std::log(arma::mean(arma::exp(llik_AB_obs.col(i))));
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     pwaic = pwaic + arma::var(llik_A_obs.col(i));
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     pwaic = pwaic + arma::var(llik_B_obs.col(i));
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     pwaic = pwaic + arma::var(llik_AB_obs.col(i));
//   }
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   return waic;
// }

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
      for(int j = 0; j < X.n_elem; j++){
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
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
               basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
               basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
                basis_funct_AB(i,0), burnin_prop, end_time);
  }
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
    }
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
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
  for(int i = 0; i < x.n_elem; i++){
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
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = calc_loglikelihood_IGP(X_AB(i,0), theta_AB, basis_coef_AB, 
            basis_funct_AB(i,0), burnin_prop, end_time);
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
  double lppd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
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
  
  for(int i = 0; i < n_AB.n_elem; i++){
    llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num);
  }
  
  // calculate log-likelihood for A
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  Rcpp::List output;
  
  // calculate log-likelihood for AB
  for(int i = 0; i < n_AB.n_elem; i++){
    calc_loglikelihood_AB_Marginal(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
                                   basis_funct_AB(i,0), burnin_prop,
                                   llik_AB(i,0), end_time);
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
      llik_AB_obs(i, j) = llik_AB(j,0)(i);
    }
  }
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    lppd = lppd + calc_log_mean(llik_AB_obs.col(i));
  }
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
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


// inline void calc_loglikelihood_AB_joint(const arma::vec X_AB,
//                                         const arma::mat theta,
//                                         const arma::field<arma::mat> Labels,
//                                         const arma::mat basis_coef_A,
//                                         const arma::mat basis_coef_B,
//                                         const arma::mat basis_funct_AB,
//                                         const int obs_num,
//                                         const double burnin_prop,
//                                         arma::mat& llik){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   arma::field<arma::mat> ph;
//   arma::vec theta_i = arma::zeros(theta.n_cols);
//   arma::vec basis_coef_A_i = arma::zeros(basis_coef_A.n_cols);
//   arma::vec basis_coef_B_i = arma::zeros(basis_coef_B.n_cols);
//   for(int i = burnin_num; i < n_MCMC; i++){
//     theta_i = theta.row(i).t();
//     basis_coef_A_i = basis_coef_A.row(i).t();
//     basis_coef_B_i = basis_coef_B.row(i).t();
//     for(int j = 0; j < X_AB.n_elem; j++){
//       if(Labels(obs_num, 0)(i,j) == 0){
//         // label is A
//         if(j != 0){
//           if(Labels(obs_num,0)(i,j-1) == 0){
//             // Condition if spike has not switched (still in A)
//             llik(i - burnin_num,j) = pinv_gauss(X_AB(j) - theta_i(4), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))),
//                                                      std::pow((1 / theta_i(3)), 2.0)) +
//                                                        dinv_gauss(X_AB(j), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0));
//           }else{
//             // Condition if spike has switched from B to A
//             llik(i - burnin_num,j) = pinv_gauss(X_AB(j), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0)) +
//               dinv_gauss(X_AB(j) - theta_i(4), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0));
//           }
//         }else{
//           llik(i - burnin_num,j) = pinv_gauss(X_AB(j), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0)) +
//             dinv_gauss(X_AB(j), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0));
//         }
//       }else{
//         // label is B
//         if(j != 0){
//           if(Labels(obs_num,0)(i, j-1) == 1){
//             // Condition if spike has not switched (still in B)
//             llik(i - burnin_num,j) = pinv_gauss(X_AB(j) - theta_i(4), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0)) +
//               dinv_gauss(X_AB(j), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0));
//           }else{
//             // Condition if spike has switched from A to B
//             llik(i - burnin_num,j) = pinv_gauss(X_AB(j), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0)) +
//               dinv_gauss(X_AB(j) - theta_i(4), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0));
//           }
//         }else{
//           llik(i - burnin_num,j) = pinv_gauss(X_AB(j), (1 / (theta_i(0) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_A_i)))), std::pow((1 / theta_i(2)), 2.0)) +
//             dinv_gauss(X_AB(j), (1 / (theta_i(1) * std::exp(arma::dot(basis_funct_AB.row(j), basis_coef_B_i)))), std::pow((1 / theta_i(3)), 2.0));
//         }
//       }
//     }
//   }
// }


// inline Rcpp::List calc_WAIC_competition_Marginal_Observation(const arma::field<arma::vec> X_A,
//                                                              const arma::field<arma::vec> X_B,
//                                                              const arma::field<arma::vec> X_AB,
//                                                              const arma::vec n_A,
//                                                              const arma::vec n_B,
//                                                              const arma::vec n_AB,
//                                                              const arma::mat theta,
//                                                              const arma::mat basis_coef_A,
//                                                              const arma::mat basis_coef_B,
//                                                              const arma::field<arma::mat> basis_funct_A,
//                                                              const arma::field<arma::mat> basis_funct_B,
//                                                              const arma::field<arma::mat> basis_funct_AB,
//                                                              const double burnin_prop){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop);
//   }
//   
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     calc_loglikelihood_AB_Marginal_Observation(X_AB(i,0), theta, basis_coef_A, basis_coef_B,
//                                                basis_funct_AB(i,0), burnin_prop,
//                                                llik_AB(i,0));
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
//     }
//   }
// 
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }

// inline Rcpp::List calc_WAIC_competition_joint(const arma::field<arma::vec> X_A,
//                                               const arma::field<arma::vec> X_B,
//                                               const arma::field<arma::vec> X_AB,
//                                               const arma::vec n_A,
//                                               const arma::vec n_B,
//                                               const arma::vec n_AB,
//                                               const arma::mat theta,
//                                               const arma::mat basis_coef_A,
//                                               const arma::mat basis_coef_B,
//                                               const arma::field<arma::mat> basis_funct_A,
//                                               const arma::field<arma::mat> basis_funct_B,
//                                               const arma::field<arma::mat> basis_funct_AB,
//                                               const arma::field<arma::mat> Labels,
//                                               const double burnin_prop,
//                                               const double& end_time){
//   int n_MCMC = theta.n_rows;
//   int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   arma::field<arma::mat> llik_AB(n_AB.n_elem, 1);
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     llik_AB(i,0) = arma::zeros(n_MCMC - burnin_num, n_AB(i));
//   }
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_A(X_A(i,0), theta, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop, end_time);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_B(X_B(i,0), theta, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop, end_time);
//   }
//   
//   
//   Rcpp::List output;
//   
//   // calculate log-likelihood for AB
//   for(int i = 0; i < n_AB.n_elem; i++){
//     calc_loglikelihood_AB_joint(X_AB(i,0), theta, Labels, basis_coef_A, basis_coef_B,
//                                 basis_funct_AB(i,0), i, burnin_prop, llik_AB(i,0));
//     Rcpp::Rcout << "Calculated loglikelihood for observation " << i << "\n";
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_AB(i,0).col(j))));
//     }
//   }
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + (arma::var(llik_A(i,0).col(j)));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + (arma::var(llik_B(i,0).col(j)));
//     }
//   }
//   
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       pwaic = pwaic + arma::var(llik_AB(i,0).col(j));
//     }
//   }
//   
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B),
//                                           Rcpp::Named("llik_AB", llik_AB));
//   return output1;
// }



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
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for joint
  for(int i = 0; i < n_joint.n_elem; i++){
    llik_joint(i,0) = calc_loglikelihood_IGP(X_joint(i,0), theta_joint, basis_coef_joint, 
            basis_funct_joint(i,0), burnin_prop, end_time);
  }
  
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  
  arma::mat llik_joint_obs = arma::zeros(n_MCMC_joint - burnin_num_joint, n_joint.n_elem);
  
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
  
  for(int i = 0; i < n_MCMC_joint - burnin_num_joint; i++){
    for(int j = 0; j < n_joint.n_elem; j++){
      llik_joint_obs(i, j) = arma::accu(llik_joint(j,0).row(i));
    }
  }
  
  // calculate log pointwise predictive density
  double lppd_seperate = 0;
  
  for(int i = 0; i < n_A.n_elem; i++){
    lppd_seperate = lppd_seperate + calc_log_mean(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
    lppd_seperate = lppd_seperate + calc_log_mean(llik_B_obs.col(i));
  }
 
  
  double lppd_joint = 0;
  for(int i = 0; i < n_joint.n_elem; i++){
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
  double var = 0;
  double mean = 0;
  double lambda;
  for(int i = 0; i < n.n_elem; i++){
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
  for(int i = 0; i < n_mean_sim.n_elem; i++){
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


// inline Rcpp::List calc_WAIC_IGP_WTA(const arma::field<arma::vec> X_A,
//                                     const arma::field<arma::vec> X_B,
//                                     const arma::vec n_A,
//                                     const arma::vec n_B,
//                                     const double& end_time,
//                                     const arma::mat theta_A,
//                                     const arma::mat basis_coef_A,
//                                     const arma::mat theta_B,
//                                     const arma::mat basis_coef_B,
//                                     const arma::field<arma::mat> basis_funct_A,
//                                     const arma::field<arma::mat> basis_funct_B,
//                                     const double burnin_prop){
//   
//   // Placeholder for log-likelihood by observation
//   arma::field<arma::mat> llik_A(n_A.n_elem, 1); 
//   arma::field<arma::mat> llik_B(n_B.n_elem, 1);
//   
//   // calculate log-likelihood for A
//   for(int i = 0; i < n_A.n_elem; i++){
//     llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
//            basis_funct_A(i,0), burnin_prop, end_time);
//   }
//   
//   // calculate log-likelihood for B
//   for(int i = 0; i < n_B.n_elem; i++){
//     llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
//            basis_funct_B(i,0), burnin_prop, end_time);
//   }
//   
//   // calculate log pointwise predictive density
//   double lppd = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_A(i,0).col(j))));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       lppd = lppd + std::log(arma::mean(arma::exp(llik_B(i,0).col(j))));
//     }
//   }
//   
//   Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
//   
//   double pwaic = 0;
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       pwaic = pwaic + arma::var(llik_A(i,0).col(j));
//     }
//   }
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       pwaic = pwaic + arma::var(llik_B(i,0).col(j));
//     }
//   }
// 
//   Rcpp::Rcout << "Effective number of parameters = " << pwaic << "\n";
//   double waic = -2 * (lppd - pwaic);
//   
//   Rcpp::Rcout << "WAIC (on deviance scale) = " << waic;
//   
//   Rcpp::List output1 = Rcpp::List::create(Rcpp::Named("WAIC", waic),
//                                           Rcpp::Named("lppd", lppd),
//                                           Rcpp::Named("Effective_pars", pwaic),
//                                           Rcpp::Named("llik_A", llik_A),
//                                           Rcpp::Named("llik_B", llik_B));
//   return output1;
// }

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
  for(int i = 0; i < n_A.n_elem; i++){
    llik_A(i,0) = calc_loglikelihood_IGP(X_A(i,0), theta_A, basis_coef_A, 
           basis_funct_A(i,0), burnin_prop, end_time);
  }
  
  // calculate log-likelihood for B
  for(int i = 0; i < n_B.n_elem; i++){
    llik_B(i,0) = calc_loglikelihood_IGP(X_B(i,0), theta_B, basis_coef_B, 
           basis_funct_B(i,0), burnin_prop, end_time);
  }
  
  arma::mat llik_A_obs = arma::zeros(n_MCMC_A - burnin_num_A, n_A.n_elem);
  arma::mat llik_B_obs = arma::zeros(n_MCMC_B - burnin_num_B, n_B.n_elem);
  
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
  
  // calculate log pointwise predictive density
  double lppd = 0;
  for(int i = 0; i < n_A.n_elem; i++){
      lppd = lppd + calc_log_mean(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
      lppd = lppd + calc_log_mean(llik_B_obs.col(i));
  }
  
  Rcpp::Rcout << "log pointwise predictive density = " << lppd << "\n";
  
  double pwaic = 0;
  for(int i = 0; i < n_A.n_elem; i++){
      pwaic = pwaic + arma::var(llik_A_obs.col(i));
  }
  for(int i = 0; i < n_B.n_elem; i++){
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