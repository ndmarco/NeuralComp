#ifndef NeuralComp_Priors_H
#define NeuralComp_Priors_H

#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
#include <mvnorm.h>

namespace NeuralComp {

// transform the parameters into an unbounded space
// theta: (I_A, I_B, sigma_A, sigma_B, delta)
inline arma::vec transform_pars(arma::vec& theta){
  arma::vec trans_theta(theta.n_elem, arma::fill::zeros);
  for(int i = 0; i < theta.n_elem; i++){
    trans_theta(i) = std::exp(theta(i));
  }
  return trans_theta;
}

// transforms the parameters back into the original space
// theta: (I_A, I_B, sigma_A, sigma_B, delta)
inline arma::vec inv_transform_pars(arma::vec& theta){
  arma::vec invtrans_theta(theta.n_elem, arma::fill::zeros);
  for(int i = 0; i < theta.n_elem; i++){
    invtrans_theta(i) = std::log(theta(i));
  }
  return invtrans_theta;
}


// Returns log pdf of inverse Gaussian distribution
// x: random variable
// mean: mean of inverse Gaussian
// shape: shape of inverse Gaussian
inline double dinv_gauss(const double x,
                         const double& mean,
                         const double& shape){
  double lpdf = -1 * INFINITY;
  if(x > 0){
    lpdf = 0.5 * std::log(shape) - 0.5 * std::log(arma::datum::pi * 2 * pow(x, 3.0)) - 
      ((shape * pow(x - mean, 2.0)) / (2 * pow(mean, 2.0) * x));
  }
  
  return lpdf;
}

// return upper probability
inline double pinv_gauss(const double x,
                         const double& mean,
                         const double& shape){
  double lcdf = 0;
  if(x > 0){
    lcdf = R::pnorm((sqrt(shape / x) * ((x/ mean) - 1)), 0, 1, true, false) + (std::exp((2 * shape) / mean) 
                                                                                 * (R::pnorm(-(sqrt(shape / x) * ((x/ mean) + 1)), 0, 1, true, false)));
    lcdf = std::log(1 -lcdf);
  }
  
  return lcdf;
}

// return sample from inverse gaussian
inline double rinv_gauss(const double mean,
                         const double shape){
  double nu = R::rnorm(0,1);
  double y = nu * nu;
  double x = mean + ((mean * mean * y) / (2 * shape)) - ((mean / (2 * shape)) * sqrt(4 * mean * shape * y + (mean * mean * y * y)));
  double z = R::runif(0,1);
  double val = (mean * mean) / x;
  if(z <= (mean / (mean + x))){
    val = x;
  }
  return val;
}

inline arma::vec rmutlinomial(const arma::vec& prob){
  arma::vec output = arma::zeros(prob.n_elem);
  int S = 1;
  double rho = 1;
  for(int i = 0; i < (prob.n_elem - 1); i++){
    if(rho > 0){
      output(i) = R::rbinom(S, prob(i) / rho);
    }
    S = S - output(i);
    rho = rho - prob(i);
  }
  output(prob.n_elem - 1) = S;
  return output;
}

// Calculate multivariate normal logpdf using precision matrix
inline double lpdf_mvnorm(const arma::vec x,
                          const arma::vec mu,
                          const arma::mat Precision){
  double lpdf = - ((x.n_elem / 2) * (std::log(2 * arma::datum::pi)  - log_det_sympd(Precision))) - 
    (0.5 * arma::dot((x - mu), Precision * (x - mu)));
  return lpdf;
}  

// calculate the log prior
// I_A_shape: shape parameter for I_A
// I_A_rate: rate parameter for I_A
// I_B_shape: shape parameter for I_B
// I_B_rate: rate parameter for I_B
// sigma_A_mean: mean parameter for sigma_A
// sigma_A_shape: shape parameter for sigma_A
// sigma_B_mean: mean parameter for sigma_B
// sigma_B_shape: shape parameter for sigma_B
// delta_shape: shape parameter for delta
// delta_rate: rate parameter for delta
// theta: (I_A, I_B, sigma_A, sigma_B, delta)
inline double log_prior_TI(const double& mu_A, 
                           const double& mu_B,
                           const double& I_A_sigma_sq,
                           const double& I_B_sigma_sq,
                           const double& sigma_A_mean,
                           const double& sigma_A_shape,
                           const double& sigma_B_mean,
                           const double& sigma_B_shape,
                           const arma::mat P_mat,
                           arma::vec& theta,
                           arma::vec& basis_coef_A,
                           arma::vec& basis_coef_B){
  
  
  
  // I_A prior
  double l_prior =  ((0.5 / I_A_sigma_sq) * arma::dot((basis_coef_A - mu_A * arma::ones(P_mat.n_rows)), P_mat * (basis_coef_A - mu_A * arma::ones(P_mat.n_rows))));
  
  // I_B prior
  l_prior = l_prior + ((0.5 / I_B_sigma_sq) * arma::dot((basis_coef_B - mu_B * arma::ones(P_mat.n_rows)), P_mat * (basis_coef_B - mu_B * arma::ones(P_mat.n_rows))));
  
  // sigma_A prior
  l_prior = l_prior + dinv_gauss(theta(0), sigma_A_mean, sigma_A_shape);
  
  // sigma_B prior
  l_prior = l_prior + dinv_gauss(theta(1), sigma_B_mean, sigma_B_shape);
  
  return l_prior;
}



// calculate the log prior
// I_A_shape: shape parameter for I_A
// I_A_rate: rate parameter for I_A
// I_B_shape: shape parameter for I_B
// I_B_rate: rate parameter for I_B
// sigma_A_mean: mean parameter for sigma_A
// sigma_A_shape: shape parameter for sigma_A
// sigma_B_mean: mean parameter for sigma_B
// sigma_B_shape: shape parameter for sigma_B
// delta_shape: shape parameter for delta
// delta_rate: rate parameter for delta
// theta: (I_A, I_B, sigma_A, sigma_B, delta)
inline double log_prior(const double& I_A_shape, 
                        const double& I_A_rate,
                        const double& I_B_shape,
                        const double& I_B_rate,
                        const double& sigma_A_mean,
                        const double& sigma_A_shape,
                        const double& sigma_B_mean,
                        const double& sigma_B_shape,
                        arma::vec& theta){
  // I_A prior
  double l_prior =  R::dgamma(theta(0), I_A_shape, (1 / I_A_rate), true);
  
  // I_B prior
  l_prior = l_prior + R::dgamma(theta(1), I_B_shape, (1 / I_B_rate), true);
  
  // sigma_A prior
  l_prior = l_prior + dinv_gauss(theta(2), sigma_A_mean, sigma_A_shape);
  
  // sigma_B prior
  l_prior = l_prior + dinv_gauss(theta(3), sigma_B_mean, sigma_B_shape);
  
  return l_prior;
}

// calculate the log prior for delta
// delta_shape: shape parameter for delta
// delta_rate: rate parameter for delta
inline double log_prior_delta(const double& delta_shape,
                              const double& delta_rate,
                              arma::vec& theta){
  // delta prior
  double l_prior = R::dgamma(theta(4), delta_shape, (1 / delta_rate), true);
  
  return l_prior;
}


}

#endif