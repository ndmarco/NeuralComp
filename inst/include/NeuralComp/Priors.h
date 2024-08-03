#ifndef NeuralComp_Priors_H
#define NeuralComp_Priors_H

#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>
#include <mvnorm.h>
#include <CppAD.h>
using namespace CppAD;
using namespace Eigen;

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
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lpdf = 0.5 * std::log(shape) - 0.5 * std::log(arma::datum::pi * 2 * std::pow(x, 3.0)) - 
          ((shape * std::pow(x - mean, 2.0)) / (2 * std::pow(mean, 2.0) * x));
      }
    }
  }
  
  return lpdf;
}

typedef AD<double> a_double;
typedef Matrix<a_double, Eigen::Dynamic, 1> a_vector;

inline a_double dinv_gauss(const double x,
                           const a_double mean,
                           const a_double shape){
  a_double lpdf = -1 * INFINITY;
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lpdf = 0.5 * CppAD::log(shape) - 0.5 * std::log(arma::datum::pi * 2 * std::pow(x, 3.0)) - 
          ((shape * (x - mean) * (x - mean) ) / (2 * (mean) * (mean) * x));
      }
    }
  }
  return lpdf;
}

inline a_double dinv_gauss(const a_double x,
                           const a_double mean,
                           const a_double shape){
  a_double lpdf = -1 * INFINITY;
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lpdf = 0.5 * CppAD::log(shape) - 0.5 * CppAD::log(arma::datum::pi * 2 * (x * x * x)) - 
          ((shape * (x - mean) * (x - mean) ) / (2 * (mean) * (mean) * x));
      }
    }
  }
  
  return lpdf;
}

// return upper probability
inline double pinv_gauss(const double x,
                         const double& mean,
                         const double& shape){
  double lcdf = 0;
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lcdf = R::pnorm((sqrt(shape / x) * ((x/ mean) - 1)), 0, 1, true, false) + (std::exp((2 * shape) / mean) 
                                                                                     * (R::pnorm(-(sqrt(shape / x) * ((x/ mean) + 1)), 0, 1, true, false)));
        lcdf = std::log(1 -lcdf);
      }
    }
  }
  
  return lcdf;
}

inline a_double pnorm_AD(const a_double x){
  a_double x1 =  x / std::sqrt(2);
  a_double pdf = 0.5 * (1 + CppAD::erf(x1));
  return pdf;
}

inline a_double pinv_gauss(const double x,
                           const a_double mean,
                           const a_double shape){
  a_double lcdf = 0;
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lcdf = pnorm_AD((sqrt(shape / x) * ((x/ mean) - 1))) + (CppAD::exp((2 * shape) / mean) 
                                                                  * (pnorm_AD(-(sqrt(shape / x) * ((x/ mean) + 1)))));
        lcdf = CppAD::log(1 -lcdf);
      }
    }
  }
  return lcdf;
}

inline a_double pinv_gauss(const a_double x,
                           const a_double mean,
                           const a_double shape){
  a_double lcdf = 0;
  if(mean > 0){
    if(shape > 0){
      if(x > 0){
        lcdf = pnorm_AD((sqrt(shape / x) * ((x/ mean) - 1))) + (CppAD::exp((2 * shape) / mean) 
                                                                  * (pnorm_AD(-(sqrt(shape / x) * ((x/ mean) + 1)))));
        lcdf = CppAD::log(1 -lcdf);
      }
    }
  }
  return lcdf;
}

// quantile function for inverse gaussian distribution
inline double qinv_gauss(const double p,
                         const double mean,
                         const double shape){
  // initialize starting point
  double m = mean * (std::sqrt((1 + std::pow((3 * mean) / (2 * shape), 2.0))) - ((3 * mean) / (2 * shape)));
  if((1 - pinv_gauss(m, mean, shape)) < 0.00001){
    double gamma_scale = (mean * mean) / shape;
    double gamma_shape =  mean / gamma_scale;
    m = R::qgamma(p, gamma_shape, gamma_scale, true, false);
  }
  
  // Newton's Method
  int i = 0;
  double delta = p - (1 - std::exp(pinv_gauss(m, mean, shape)));
  double movement = 1;
  while ((i < 100) && (std::abs(movement) > 1e-10)){
    if(delta < 1e-5){
      movement = (delta / std::exp(dinv_gauss(m, mean, shape)));
      m = m + movement;
    }else{
      movement = (delta * std::exp(std::log(p) + std::log(1 - (delta / 2)))) / std::exp(dinv_gauss(m, mean, shape));
      m = m + movement;
    }
    delta = p - (1 - std::exp(pinv_gauss(m, mean, shape)));
    i = i + 1;
  }
  
  return m;
}

inline double dinv_gauss_trunc(const double x,
                               const double mean,
                               const double shape,
                               const double lower_trunc,
                               const double upper_trunc){
  double p_lower = 0;
  double p_upper = 1;
  if(lower_trunc > 0){
    p_lower = 1 - std::exp(pinv_gauss(lower_trunc, mean, shape));
  }
  if(upper_trunc != INFINITY){
    p_upper = 1 - std::exp(pinv_gauss(upper_trunc, mean, shape));
  }
  double log_p_diff = std::log(p_upper - p_lower);
  double lpdf = dinv_gauss(x, mean, shape) - log_p_diff;
  if(x < lower_trunc){
    lpdf = -1 * INFINITY;
  }
  if(x > upper_trunc){
    lpdf = -1 * INFINITY;
  }
  return lpdf;
}
  
  
inline double rinv_gauss_trunc(const double mean,
                               const double shape,
                               const double lower_trunc,
                               const double upper_trunc){
  double u = R::runif(0,1);
  double p_lower = 0;
  double p_upper = 1;
  if(lower_trunc > 0){
    p_lower = 1 - std::exp(pinv_gauss(lower_trunc, mean, shape));
  }
  if(upper_trunc != INFINITY){
    p_upper = 1 - std::exp(pinv_gauss(upper_trunc, mean, shape));
  }
  double p_comb = p_lower + u*(p_upper - p_lower);
  return qinv_gauss(p_comb, mean, shape);
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
   // check for overflow
  if(val < 0){
    val = mean;
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
inline double log_prior_FR(const double& I_A_sigma_sq,
                           const double& I_B_sigma_sq,
                           const arma::vec theta,
                           arma::vec& basis_coef_A,
                           arma::vec& basis_coef_B){
  
  // I_A prior
  double l_prior =  -((0.5 / I_A_sigma_sq) * arma::dot(basis_coef_A, basis_coef_A)) - (basis_coef_A.n_elem * std::log(std::sqrt(I_A_sigma_sq * 2 * arma::datum::pi)));
  
  // I_B prior
  l_prior = l_prior - ((0.5 / I_B_sigma_sq) * arma::dot(basis_coef_B , basis_coef_B)) - (basis_coef_B.n_elem * std::log(std::sqrt(I_B_sigma_sq * 2 * arma::datum::pi)));
  
  return l_prior;
}

inline double log_prior_FR1(const double& I_sigma_sq,
                            const arma::vec theta,
                            arma::vec& basis_coef){
  // I_A prior
  double l_prior =  -((0.5 / I_sigma_sq) * arma::dot(basis_coef, basis_coef))  - (basis_coef.n_elem * std::log(std::sqrt(I_sigma_sq * 2 * arma::datum::pi)));
  

  return l_prior;
}


inline a_double dot_AD(arma::rowvec basis_func,
                       a_vector basis_coef,
                       bool Indicator_A){
  a_double output = 0;
  for(int i = 0; i < basis_func.n_elem; i++){
    if(Indicator_A == true){
      output = output + (basis_func(i) * basis_coef(i));
    }else{
      output = output + (basis_func(i) * basis_coef(i + (basis_coef.rows() / 2)));
    }
  }
  return output;
}

inline a_double dot_AD1(arma::rowvec basis_func,
                       a_vector basis_coef){
  a_double output = 0;
  for(int i = 0; i < basis_func.n_elem; i++){
    output = output + (basis_func(i) * basis_coef(i));
  }
  return output;
}

inline a_double norm_AD(a_vector basis_coef,
                       bool Indicator_A){
  a_double output = 0;
  for(int i = 0; i < (basis_coef.rows() / 2); i++){
    if(Indicator_A == true){
      output = output + (basis_coef(i) * basis_coef(i));
    }else{
      output = output + (basis_coef(i + (basis_coef.rows() / 2)) * basis_coef(i + (basis_coef.rows() / 2)));
    }
  }
  return output;
}

inline a_double log_prior_FR(const double& I_A_sigma_sq,
                             const double& I_B_sigma_sq,
                             arma::vec theta,
                             a_vector basis_coef){
  
  // I_A prior
  a_double l_prior =  -((0.5 / I_A_sigma_sq) * norm_AD(basis_coef, true)) - ((basis_coef.rows() / 2) * std::log(std::sqrt(I_A_sigma_sq * 2 * arma::datum::pi)));
  
  // I_B prior
  l_prior = l_prior - ((0.5 / I_B_sigma_sq) * norm_AD(basis_coef, false)) - ((basis_coef.rows() / 2) * std::log(std::sqrt(I_B_sigma_sq * 2 * arma::datum::pi)));

  
  return l_prior;
}


inline a_double norm_AD1(a_vector basis_coef){
  a_double output = 0;
  for(int i = 0; i < basis_coef.rows(); i++){
      output = output + (basis_coef(i) * basis_coef(i));
  }
  return output;
} 

inline a_double log_prior_FR1(const double& I_sigma_sq,
                              arma::vec theta,
                              a_vector basis_coef){
  
  // I_A prior
  a_double l_prior =  -((0.5 / I_sigma_sq) * norm_AD1(basis_coef)) - (basis_coef.rows() * std::log(std::sqrt(I_sigma_sq * 2 * arma::datum::pi)));

  
  return l_prior;
}

// // calculate the log prior
// // I_A_shape: shape parameter for I_A
// // I_A_rate: rate parameter for I_A
// // I_B_shape: shape parameter for I_B
// // I_B_rate: rate parameter for I_B
// // sigma_A_mean: mean parameter for sigma_A
// // sigma_A_shape: shape parameter for sigma_A
// // sigma_B_mean: mean parameter for sigma_B
// // sigma_B_shape: shape parameter for sigma_B
// // delta_shape: shape parameter for delta
// // delta_rate: rate parameter for delta
// // theta: (I_A, I_B, sigma_A, sigma_B, delta)
// inline double log_prior_sigma(const double& sigma_A_mean,
//                               const double& sigma_A_shape,
//                               const double& sigma_B_mean,
//                               const double& sigma_B_shape,
//                               arma::vec& theta){
//   
//   // sigma_A prior
//   double  l_prior = dinv_gauss(theta(0), sigma_A_mean, sigma_A_shape);
//   
//   // sigma_B prior
//   l_prior = l_prior + dinv_gauss(theta(1), sigma_B_mean, sigma_B_shape);
//   
//   return l_prior;
// }


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
inline double log_prior(const double& I_A_mean, 
                        const double& I_A_shape,
                        const double& I_B_mean,
                        const double& I_B_shape,
                        const double& sigma_A_mean,
                        const double& sigma_A_shape,
                        const double& sigma_B_mean,
                        const double& sigma_B_shape,
                        const arma::vec basis_coef_A,
                        const arma::vec basis_coef_B,
                        arma::vec& theta){
  
  // I_A prior
  double l_prior =   (dinv_gauss(theta(0), I_A_mean, I_A_shape));
  
  // I_B prior
  l_prior = l_prior + (dinv_gauss(theta(1), I_B_mean, I_B_shape));
  
  // sigma_A prior
  l_prior = l_prior + dinv_gauss(theta(2), sigma_A_mean, sigma_A_shape);
  
  // sigma_B prior
  l_prior = l_prior + dinv_gauss(theta(3), sigma_B_mean, sigma_B_shape);
  
  
  return l_prior;
}


inline a_double log_prior(const double& I_A_mean, 
                          const double& I_A_shape,
                          const double& I_B_mean,
                          const double& I_B_shape,
                          const double& sigma_A_mean,
                          const double& sigma_A_shape,
                          const double& sigma_B_mean,
                          const double& sigma_B_shape,
                          const arma::vec basis_coef_A,
                          const arma::vec basis_coef_B,
                          a_vector theta){
  
  // I_A prior
  a_double l_prior = (dinv_gauss(theta(0), I_A_mean, I_A_shape));
  
  // I_B prior
  l_prior = l_prior + (dinv_gauss(theta(1), I_B_mean, I_B_shape));
  
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