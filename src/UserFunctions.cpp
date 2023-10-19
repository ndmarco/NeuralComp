#include <RcppArmadillo.h>
#include <cmath>
#include <NeuralComp.h>


//' HMC sampler for competition model
//' 
//' @name HMC
//' @param Labels List of vectors containing labels containing membership of spike for AB trials
//' @param X_A List of vectors containing ISIs for A trials
//' @param X_B List of vectors containing ISIs for B trials
//' @param X_AB List of vectors containing ISIs for AB trials
//' @param n_A Vector containing number of spikes per trial for A stimulus trials
//' @param n_B Vector containing number of spikes per trial for B stimulus trials
//' @param n_AB Vector containing number of spikes per trial for AB stimulus trials
//' @param MCMC_iters Integer containing number of HMC iterations
//' @param init_position Vector containing starting position for HMC
//' @param Leapfrog_steps Integer containing number of leapfrog steps for approximating the Hamiltonian dynamics
//' @param I_A_shape Double containing shape parameter for prior distribution on I_A
//' @param I_A_rate Double containing rate parameter for prior distribution on I_A
//' @param I_B_shape Double containing shape parameter for prior distribution on I_B
//' @param I_B_rate Double containing rate parameter for prior distribution on I_B
//' @param sigma_A_mean Double containing mean parameter for prior distribution on sigma_A
//' @param sigma_A_shape Double containing shape parameter for prior distribution on sigma_A
//' @param sigma_B_mean Double containing mean parameter for prior distribution on sigma_B
//' @param sigma_B_shape Double containing shape parameter for prior distribution on sigma_B
//' @param delta_shape Double containing shape parameter for prior distribution on delta
//' @param delta_rate Double containing rate parameter for prior distribution on delta
//' @param eps_step Vector containing step size for approximating gradient
//' @param step_size Vector containing step size for leapfrog integrator
//' 
//' @export
//[[Rcpp::export]]
arma::mat HMC(arma::field<arma::vec> Labels,
              const arma::field<arma::vec> X_A,
              const arma::field<arma::vec> X_B,
              const arma::field<arma::vec> X_AB,
              const arma::vec n_A,
              const arma::vec n_B,
              const arma::vec n_AB,
              int MCMC_iters,
              Rcpp::Nullable<Rcpp::NumericVector> init_position = R_NilValue,
              int Leapfrog_steps = 10,
              const double I_A_shape = 40, 
              const double I_A_rate = 1,
              const double I_B_shape = 40,
              const double I_B_rate = 1,
              const double sigma_A_mean = 6.32,
              const double sigma_A_shape = 1,
              const double sigma_B_mean = 6.32,
              const double sigma_B_shape = 1,
              const double delta_shape = 0.5,
              const double delta_rate = 0.1,
              Rcpp::Nullable<Rcpp::NumericVector> eps_step = R_NilValue,
              Rcpp::Nullable<Rcpp::NumericVector> step_size =  R_NilValue){
  arma::vec init_position1;
  if(init_position.isNull()){
    init_position1 = {40, 40, sqrt(40), sqrt(40), 0.001};
  }else{
    Rcpp::NumericMatrix X_(init_position);
    init_position1 = Rcpp::as<arma::mat>(X_);
  }
  arma::vec eps_step1;
  if(eps_step.isNull()){
    eps_step1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericMatrix X_(eps_step);
    eps_step1 = Rcpp::as<arma::mat>(X_);
  }
  arma::vec step_size1;
  if(step_size.isNull()){
    step_size1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericMatrix X_(step_size);
    step_size1 = Rcpp::as<arma::mat>(X_);
  }
  
  
  arma::mat theta = NeuralComp::HMC_sampler(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, init_position1,
                                            MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                            I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                            sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                            eps_step1, step_size1);
   return theta;
   
}


// //[[Rcpp::export]]
// arma::vec trans_calc_gradient1(arma::field<arma::vec>& Labels,
//                                arma::vec& theta,
//                                const arma::field<arma::vec>& X_A,
//                                const arma::field<arma::vec>& X_B,
//                                const arma::field<arma::vec>& X_AB,
//                                const arma::vec& n_A,
//                                const arma::vec& n_B,
//                                const arma::vec& n_AB,
//                                const double& I_A_shape, 
//                                const double& I_A_rate,
//                                const double& I_B_shape,
//                                const double& I_B_rate,
//                                const double& sigma_A_mean,
//                                const double& sigma_A_shape,
//                                const double& sigma_B_mean,
//                                const double& sigma_B_shape,
//                                const double& delta_shape,
//                                const double& delta_rate,
//                                const arma::vec& eps_step){
//   
//   arma::vec grad = NeuralComp::calc_gradient(Labels, NeuralComp::transform_pars(theta), X_A, X_B, X_AB, n_A,
//                                  n_B, n_AB, I_A_shape, I_A_rate, I_B_shape, I_B_rate,
//                                  sigma_A_mean, sigma_A_shape, sigma_B_mean, sigma_B_shape,
//                                  delta_shape, delta_rate, eps_step);
//   grad = grad + arma::ones(grad.n_elem);
//   
//   return(grad);
// }
// 
// 
// //[[Rcpp::export]]
// arma::vec calc_gradient1(arma::field<arma::vec>& Labels,
//                          arma::vec theta,
//                          const arma::field<arma::vec>& X_A,
//                          const arma::field<arma::vec>& X_B,
//                          const arma::field<arma::vec>& X_AB,
//                          const arma::vec& n_A,
//                          const arma::vec& n_B,
//                          const arma::vec& n_AB,
//                          const double& I_A_shape, 
//                          const double& I_A_rate,
//                          const double& I_B_shape,
//                          const double& I_B_rate,
//                          const double& sigma_A_mean,
//                          const double& sigma_A_shape,
//                          const double& sigma_B_mean,
//                          const double& sigma_B_shape,
//                          const double& delta_shape,
//                          const double& delta_rate,
//                          const arma::vec& eps_step){
//   arma::vec grad(theta.n_elem, arma::fill::zeros);
//   arma::vec theta_p_eps = theta;
//   arma::vec theta_m_eps = theta;
//   for(int i = 0; i < theta.n_elem; i++){
//     theta_p_eps = theta;
//     // f(x + e) in the i^th dimension
//     theta_p_eps(i) = theta_p_eps(i) + eps_step(i);
//     theta_m_eps = theta;
//     // f(x - e) in the i^th dimension
//     theta_m_eps(i) = theta_m_eps(i) - eps_step(i);
//     // approximate gradient ((f(x + e) f(x - e))/ 2e)
//     grad(i) = (NeuralComp::log_posterior(Labels, theta_p_eps, X_A, X_B, X_AB,
//                n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
//                I_B_rate, sigma_A_mean, sigma_A_shape,
//                sigma_B_mean, sigma_B_shape, delta_shape,
//                delta_rate) - NeuralComp::log_posterior(Labels, theta_m_eps, X_A, X_B, X_AB,
//                n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
//                I_B_rate, sigma_A_mean, sigma_A_shape,
//                sigma_B_mean, sigma_B_shape, delta_shape,
//                delta_rate)) / (2 * eps_step(i));
//     
//   }
//   return grad;
// }
// 
// //[[Rcpp::export]]
// double log_likelihood1(arma::field<arma::vec>& Labels,
//                        arma::vec& theta,
//                        const arma::field<arma::vec>& X_A,
//                        const arma::field<arma::vec>& X_B,
//                        const arma::field<arma::vec>& X_AB,
//                        const arma::vec& n_A,
//                        const arma::vec& n_B,
//                        const arma::vec& n_AB){
//   double l_likelihood = 0;
//   
//   // Calculate log-likelihood for A trials
//   for(int i = 0; i < n_A.n_elem; i++){
//     for(int j = 0; j < n_A(i); j++){
//       l_likelihood = l_likelihood + NeuralComp::dinv_gauss(X_A(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
//     }
//   }
//   
//   // Calculate log-likelihood for B trials
//   for(int i = 0; i < n_B.n_elem; i++){
//     for(int j = 0; j < n_B(i); j++){
//       l_likelihood = l_likelihood + NeuralComp::dinv_gauss(X_B(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
//     }
//   }
//   
//   // calculate log-likelihood for AB trials
//   for(int i = 0; i < n_AB.n_elem; i++){
//     for(int j = 0; j < n_AB(i); j++){
//       if(Labels(i,0)(j) == 0){
//         // label is A
//         if(j != 0){
//           if(Labels(i,0)(j-1) == 0){
//             // Condition if spike has not switched (still in A)
//             l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
//               NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
//           }else{
//             // Condition if spike has switched from B to A
//             l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
//               NeuralComp::dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
//           }
//         }else{
//           l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
//             NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
//         }
//       }else{
//         // label is B
//         if(j != 0){
//           if(Labels(i,0)(j-1) == 1){
//             // Condition if spike has not switched (still in A)
//             l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
//               NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
//           }else{
//             // Condition if spike has switched from B to A
//             l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
//               NeuralComp::dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
//           }
//         }else{
//           l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
//             NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
//         }
//       }
//     }
//   }
//   return l_likelihood;
// }
// 
// //[[Rcpp::export]]
// double pinv_gauss1(double x,
//                    double a,
//                    double b){
//   return NeuralComp::pinv_gauss(x, a, b);
// }
// 
// 
// //[[Rcpp::export]]
// double log_posterior1(arma::field<arma::vec>& Labels,
//                      arma::vec theta,
//                      const arma::field<arma::vec>& X_A,
//                      const arma::field<arma::vec>& X_B,
//                      const arma::field<arma::vec>& X_AB,
//                      const arma::vec& n_A,
//                      const arma::vec& n_B,
//                      const arma::vec& n_AB,
//                      const double& I_A_shape, 
//                      const double& I_A_rate,
//                      const double& I_B_shape,
//                      const double& I_B_rate,
//                      const double& sigma_A_mean,
//                      const double& sigma_A_shape,
//                      const double& sigma_B_mean,
//                      const double& sigma_B_shape,
//                      const double& delta_shape,
//                      const double& delta_rate){
//   double l_posterior = NeuralComp::log_likelihood(Labels, theta, X_A, X_B, X_AB, n_A, n_B, n_AB) +
//     NeuralComp::log_prior(I_A_shape, I_A_rate, I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
//               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate, theta);
//   return l_posterior;
// }