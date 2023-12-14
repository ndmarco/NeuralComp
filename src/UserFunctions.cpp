#include <RcppArmadillo.h>
#include <cmath>
#include <NeuralComp.h>
#include <splines2Armadillo.h>
#include <CppAD.h>

using namespace CppAD;
using namespace Eigen;


//' Construct B-splines
//' 
//' Creates B-spline basis functions evaluated at the time points of interest.
//' 
//' @name getBSpline
//' @param time Vector of time points of interest
//' @param basis_degree Integer indicating the degree of B-splines
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @returns Matrix containing the B-splines evaluated at the time points of interest
//' 
//' @examples
//' time <- seq(0, 1, 0.01)
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' 
//' B <- getBSpline(time, basis_degree, boundary_knots, internal_knots)
//' 
//' @export
//[[Rcpp::export]]
arma::mat getBSpline(const arma::vec time,
                     const int basis_degree,
                     const arma::vec boundary_knots,
                     const arma::vec internal_knots){
  arma::mat basis_funct;
  splines2::BSpline bspline;
  
  bspline = splines2::BSpline(time, internal_knots, basis_degree,
                              boundary_knots);
  arma::mat bspline_mat{bspline.basis(true)};
  basis_funct = bspline_mat;
  return basis_funct;
}

//[[Rcpp::export]]
arma::mat GetTraceLabels(const arma::field<arma::vec> MCMC_output,
                         int sample_num,
                         int MCMC_iters){
  arma::mat trace(MCMC_output(sample_num-1, 0).n_elem, MCMC_iters, arma::fill::zeros);
  for(int i = 0; i < MCMC_output(sample_num-1, 0).n_elem; i++){
    for(int j = 0; j < MCMC_iters; j++){
      Rcpp::Rcout << i << "  " << j;
      trace(i, j) = MCMC_output(sample_num-1,j)(i);
    }
  }
  return trace;
}

//' Sampler for Drift-Diffusion Competition Model
//' 
//' Conducts MCMC to get posterior samples from the drift-diffusion competition model.
//' This function can fit a time-homogeneous model, as well as a time-inhomogeneous 
//' model.
//' 
//' @name Sampler_Competition
//' @param X_A List of vectors containing the ISIs of A trials
//' @param X_B List of vectors containing the ISIs of B trials
//' @param X_AB List of vectors containing the ISIs of AB trials
//' @param n_A Vector containing number of spikes for each A trial
//' @param n_B Vector containing number of spikes for each B trial
//' @param n_AB Vector containing number of spikes for each AB trial
//' @param MCMC_iters Integer containing the number of MCMC_iterations excluding warm up blocks
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param Warm_block1 Integer containing number of iterations to adapt the leapfrog step size under identity mass matrices
//' @param Warm_block2 Integer containing number of iterations to adapt the mass matrices, leapfrog step size, and delta sampling parameters
//' @param Leapfrog_steps Integer containing number of leapfrog steps per HMC step
//' @param I_A_mean Double containing the value for the mean parameter of IG prior on I_A
//' @param I_A_shape Double containing the value for the shape parameter of IG prior on I_A
//' @param I_B_mean Double containing the value for the mean parameter of IG prior on I_B
//' @param I_B_shape Double containing the value for the shape parameter of IG prior on I_B
//' @param sigma_A_mean Double containing the value for the mean parameter of IG prior on sigma_A
//' @param sigma_A_shape Double containing the value for the shape parameter of IG prior on sigma_A
//' @param sigma_B_mean Double containing the value for the mean parameter of IG prior on sigma_B
//' @param sigma_B_shape Double containing the value for the shape parameter of IG prior on sigma_B
//' @param delta_shape Double containing the value for the shape parameter of gamma prior on delta
//' @param delta_rate Double containing the value for the rate parameter of gamma prior on delta
//' @param step_size_theta Double containing initial leapfrog step size for theta parameters
//' @param step_size_FR Double containing initial leapfrog step size for time-inhomogeneous firing rate parameters
//' @param delta_proposal_mean Double containing the value for mean parameter of the lognormal proposal distribution of delta parameter
//' @param delta_proposal_sd Double containing the value for sd parameter of the lognormal proposal distribution of delta parameter
//' @param alpha_labels Double containing probability that proposed new deltas come from the prior on delta instead of adapted lognormal distribution
//' @param alpha Double containing the value for the shape parameter of the inverse gamma prior on I_A_sigma squared and I_B_sigma squared
//' @param beta Double containing the value for the scale parameter of the inverse gamma prior on I_A_sigma squared and I_B_sigma squared
//' @param delta_adaptation_block Integer containing how often the delta sampling parameters should be updated in Warm_block2
//' @param Mass_adaptation_block Integer containing how often the Mass Matrix should be updated in Warm_block2
//' @param M_proposal Integer containing the number of deltas proposed when sampling delta
//' @returns List containing:
//' \describe{
//'   \item{\code{theta}}{Matrix of samples of the theta parameters (I_A, I_B, sigma_A, sigma_B, delta) where each row is an MCMC sample}
//'   \item{\code{labels}}{List of matrices where each item in the list contains MCMC samples from an AB trial}
//'   \item{\code{basis_coef_A}}{Matrix containing MCMC samples of the coefficients of the B-splines for the A process (if time-inhomogeneous)}
//'   \item{\code{basis_coef_B}}{Matrix containing MCMC samples of the coefficients of the B-splines for the B process (if time-inhomogeneous)}
//'   \item{\code{I_A_sigma_sq}}{Vector of MCMC samples of I_A_sigma squared}
//'   \item{\code{I_B_sigma_sq}}{Vector of MCMC samples of I_B_sigma squared}
//'   \item{\code{LogLik}}{Log-likelihood plot of best performing chain}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{I_A_mean}}{must be greater than 0}
//'   \item{\code{I_A_shape}}{must be greater than 0}
//'   \item{\code{I_B_mean}}{must be greater than 0}
//'   \item{\code{I_B_shape}}{must be greater than 0}
//'   \item{\code{sigma_A_mean}}{must be greater than 0}
//'   \item{\code{sigma_A_shape}}{must be greater than 0}
//'   \item{\code{sigma_B_mean}}{must be greater than 0}
//'   \item{\code{sigma_B_shape}}{must be greater than 0}
//'   \item{\code{delta_shape}}{must be greater than 0}
//'   \item{\code{delta_rate}}{must be greater than 0}
//'   \item{\code{step_size_theta}}{must be greater than 0}
//'   \item{\code{step_size_FR}}{must be greater than 0}
//'   \item{\code{delta_proposal_sd}}{must be greater than 0}
//'   \item{\code{alpha_labels}}{must be between 0 and 1}
//'   \item{\code{alpha}}{must be greater than 0}
//'   \item{\code{beta}}{must be greater than 0}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS", package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' # Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                Warm_block1 = Warm_block1, Warm_block2 = Warm_block2,
//'                                time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS", package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' # Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                Warm_block1 = Warm_block1, Warm_block2 = Warm_block2)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List Sampler_Competition(const arma::field<arma::vec> X_A,
                               const arma::field<arma::vec> X_B,
                               const arma::field<arma::vec> X_AB,
                               const arma::vec n_A,
                               const arma::vec n_B,
                               const arma::vec n_AB,
                               int MCMC_iters,
                               const int basis_degree,
                               const arma::vec boundary_knots,
                               const arma::vec internal_knots,
                               bool time_inhomogeneous = true,
                               int Warm_block1 = 500,
                               int Warm_block2 = 1000,
                               int Leapfrog_steps = 10,
                               const double I_A_mean = 40, 
                               const double I_A_shape = 1,
                               const double I_B_mean = 40,
                               const double I_B_shape = 1,
                               const double sigma_A_mean = 6.32,
                               const double sigma_A_shape = 1,
                               const double sigma_B_mean = 6.32,
                               const double sigma_B_shape = 1,
                               const double delta_shape = 0.01,
                               const double delta_rate = 0.1,
                               double step_size_theta =  0.001,
                               double step_size_FR =  0.001,
                               double delta_proposal_mean = -2,
                               double delta_proposal_sd = 0.3,
                               double alpha_labels = 0.2,
                               double alpha = 1,
                               double beta = 0.005,
                               int delta_adaption_block = 100,
                               int Mass_adaption_block = 500,
                               int M_proposal = 10){
  Rcpp::List param;
  if(time_inhomogeneous == true){
    //Create B-splines
    splines2::BSpline bspline;
    // Make spline basis for A functions
    arma::field<arma::mat> basis_funct_A(n_A.n_elem,1);
    for(int i = 0; i < n_A.n_elem; i++){
      arma::vec time = arma::zeros(n_A(i));
      for(int j = 1; j < n_A(i); j++){
        time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(true)};
      basis_funct_A(i,0) = bspline_mat;
    }
    
    // Make spline basis for B functions
    arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
    for(int i = 0; i < n_B.n_elem; i++){
      arma::vec time = arma::zeros(n_B(i));
      for(int j = 1; j < n_B(i); j++){
        time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(true)};
      basis_funct_B(i,0) = bspline_mat;
    }
    
    // Make spline basis for AB functions
    arma::field<arma::mat> basis_funct_AB(n_AB.n_elem,1);
    for(int i = 0; i < n_AB.n_elem; i++){
      arma::vec time = arma::zeros(n_AB(i));
      for(int j = 1; j < n_AB(i); j++){
        time(j) = arma::accu(X_AB(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(true)};
      basis_funct_AB(i,0) = bspline_mat;
    }
    
    param = NeuralComp::Mixed_sampler_int_TI(basis_funct_A, basis_funct_B, basis_funct_AB,
                                             X_A, X_B, X_AB, n_A, n_B, n_AB, MCMC_iters, 
                                             Leapfrog_steps, I_A_mean, I_A_shape,
                                             I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
                                             sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                             step_size_theta, step_size_FR,
                                             delta_proposal_mean, delta_proposal_sd, 
                                             alpha_labels, alpha, beta, delta_adaption_block,
                                             Mass_adaption_block, M_proposal, 
                                             Warm_block1, Warm_block2);
  }else{
    param = NeuralComp::Mixed_sampler_int(X_A, X_B, X_AB, n_A, n_B, n_AB, MCMC_iters, 
                                          Leapfrog_steps, I_A_mean, I_A_shape,
                                          I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
                                          sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                          step_size_theta, step_size_FR,
                                          delta_proposal_mean, delta_proposal_sd, 
                                          alpha_labels, delta_adaption_block,
                                          Mass_adaption_block, M_proposal, 
                                          Warm_block1, Warm_block2);
  }
  
  return param;
}
