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
  // Check conditions
  if(basis_degree <  1){
    Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
  }
  for(int i = 0; i < internal_knots.n_elem; i++){
    if(boundary_knots(0) >= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
    }
    if(boundary_knots(1) <= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
    }
  }
  
  arma::mat basis_funct;
  splines2::BSpline bspline;
  
  bspline = splines2::BSpline(time, internal_knots, basis_degree,
                              boundary_knots);
  arma::mat bspline_mat{bspline.basis(true)};
  basis_funct = bspline_mat;
  return basis_funct;
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
//'   \item{\code{LogLik}}{Log-likelihood for each MCMC iteration}
//'   \item{\code{LogPosterior}}{Log-Posterior for each MCMC iteration}
//'   
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
//' 
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
                               int Warm_block2 = 3000,
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
 
 // Check conditions
 if(I_A_mean <= 0){
   Rcpp::stop("'I_A_mean' must be greater than 0");
 }
 if(I_A_shape <= 0){
   Rcpp::stop("'I_A_shape' must be greater than 0");
 }
 if(I_B_mean <= 0){
   Rcpp::stop("'I_B_mean' must be greater than 0");
 }
 if(I_B_shape <= 0){
   Rcpp::stop("'I_B_shape' must be greater than 0");
 }
 if(sigma_A_mean <= 0){
   Rcpp::stop("'sigma_A_mean' must be greater than 0");
 }
 if(sigma_A_shape <= 0){
   Rcpp::stop("'sigma_A_shape' must be greater than 0");
 }
 if(sigma_B_mean <= 0){
   Rcpp::stop("'sigma_B_mean' must be greater than 0");
 }
 if(sigma_B_shape <= 0){
   Rcpp::stop("'sigma_B_shape' must be greater than 0");
 }
 if(delta_rate <= 0){
   Rcpp::stop("'delta_rate' must be greater than 0");
 }
 if(delta_shape <= 0){
   Rcpp::stop("'delta_shape' must be greater than 0");
 }
 if(step_size_theta <= 0){
   Rcpp::stop("'step_size_theta' must be greater than 0");
 }
 if(step_size_FR <= 0){
   Rcpp::stop("'step_size_FR' must be greater than 0");
 }
 if(delta_proposal_sd <= 0){
   Rcpp::stop("'delta_proposal_sd' must be greater than 0");
 }
 if(alpha_labels <= 0){
   Rcpp::stop("'alpha_labels' must be between 0 and 1");
 }
 if(alpha_labels >= 1){
   Rcpp::stop("'alpha_labels' must be between 0 and 1");
 }
 if(alpha <= 0){
   Rcpp::stop("'alpha' must be between 0 and 1");
 }
 if(beta <= 0){
   Rcpp::stop("'beta' must be between 0 and 1");
 }
 
 Rcpp::List param;
 if(time_inhomogeneous == true){
   
   // Check conditions
   if(basis_degree <  1){
     Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
   }
   for(int i = 0; i < internal_knots.n_elem; i++){
     if(boundary_knots(0) >= internal_knots(i)){
       Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
     }
     if(boundary_knots(1) <= internal_knots(i)){
       Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
     }
   }
   
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
                                         step_size_theta, delta_proposal_mean, delta_proposal_sd, 
                                         alpha_labels, delta_adaption_block,
                                         Mass_adaption_block, M_proposal, 
                                         Warm_block1, Warm_block2);
 }
 
 return param;
}

//' Sampler for Inverse Gamma Renewal Process
//' 
//' Conducts MCMC to get posterior samples from an inverse Gaussian renewal process.
//' This function can fit a time-homogeneous model, as well as a time-inhomogeneous 
//' model.
//' 
//' @name Sampler_IGP
//' @param X List of vectors containing the ISIs the trials
//' @param n Vector containing number of spikes for the trials
//' @param MCMC_iters Integer containing the number of MCMC_iterations excluding warm up blocks
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param Warm_block1 Integer containing number of iterations to adapt the leapfrog step size under identity mass matrices
//' @param Warm_block2 Integer containing number of iterations to adapt the mass matrices, leapfrog step size, and delta sampling parameters
//' @param Leapfrog_steps Integer containing number of leapfrog steps per HMC step
//' @param I_mean Double containing the value for the mean parameter of IG prior on I
//' @param I_shape Double containing the value for the shape parameter of IG prior on I
//' @param sigma_mean Double containing the value for the mean parameter of IG prior on sigma
//' @param sigma_shape Double containing the value for the shape parameter of IG prior on sigma
//' @param step_size_theta Double containing initial leapfrog step size for theta parameters
//' @param step_size_FR Double containing initial leapfrog step size for time-inhomogeneous firing rate parameters
//' @param alpha Double containing the value for the shape parameter of the inverse gamma prior on I_A_sigma squared and I_B_sigma squared
//' @param beta Double containing the value for the scale parameter of the inverse gamma prior on I_A_sigma squared and I_B_sigma squared
//' @param Mass_adaptation_block Integer containing how often the Mass Matrix should be updated in Warm_block2
//' @param M_proposal Integer containing the number of deltas proposed when sampling delta
//' @returns List containing:
//' \describe{
//'   \item{\code{theta}}{Matrix of samples of the theta parameters (I, sigma) where each row is an MCMC sample}
//'   \item{\code{basis_coef}}{Matrix containing MCMC samples of the coefficients of the B-splines for the inhomogeneous process (if time-inhomogeneous)}
//'   \item{\code{I_sigma_sq}}{Vector of MCMC samples of I_sigma squared}
//'   \item{\code{LogLik}}{Log-likelihood for each MCMC iteration}
//'   \item{\code{LogPosterior}}{Log-Posterior for each MCMC iteration}
//'   
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{I_mean}}{must be greater than 0}
//'   \item{\code{I_shape}}{must be greater than 0}
//'   \item{\code{sigma_mean}}{must be greater than 0}
//'   \item{\code{sigma_shape}}{must be greater than 0}
//'   \item{\code{step_size_theta}}{must be greater than 0}
//'   \item{\code{step_size_FR}}{must be greater than 0}
//'   \item{\code{delta_proposal_sd}}{must be greater than 0}
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
//' 
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
Rcpp::List Sampler_IGP(const arma::field<arma::vec> X,
                       const arma::vec n,
                       int MCMC_iters,
                       const int basis_degree,
                       const arma::vec boundary_knots,
                       const arma::vec internal_knots,
                       bool time_inhomogeneous = true,
                       int Warm_block1 = 500,
                       int Warm_block2 = 3000,
                       int Leapfrog_steps = 10,
                       const double I_mean = 40, 
                       const double I_shape = 1,
                       const double sigma_mean = 6.32,
                       const double sigma_shape = 1,
                       double step_size_theta =  0.001,
                       double step_size_FR =  0.001,
                       double alpha = 1,
                       double beta = 0.005,
                       int Mass_adaption_block = 500,
                       int M_proposal = 10){
 Rcpp::List param;
 if(time_inhomogeneous == true){
   //Create B-splines
   splines2::BSpline bspline;
   // Make spline basis for A functions
   arma::field<arma::mat> basis_funct(n.n_elem,1);
   for(int i = 0; i < n.n_elem; i++){
     arma::vec time = arma::zeros(n(i));
     for(int j = 1; j < n(i); j++){
       time(j) = arma::accu(X(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(true)};
     basis_funct(i,0) = bspline_mat;
   }
   
   Rcpp::Rcout << "Made it ";
   param = NeuralComp::Mixed_sampler_IGP_int_TI(basis_funct, X, n, MCMC_iters, 
                                            Leapfrog_steps, I_mean, I_shape,
                                            sigma_mean, sigma_shape,
                                            step_size_theta, step_size_FR,
                                            alpha, beta, Mass_adaption_block, 
                                            Warm_block1, Warm_block2);
 }else{
   param = NeuralComp::Mixed_sampler_IGP_int(X, n, MCMC_iters, Leapfrog_steps, I_mean,
                                             I_shape, sigma_mean, sigma_shape,
                                             step_size_theta, Mass_adaption_block,
                                             Warm_block1, Warm_block2);
 }
 
 return param;
}

//' Construct CI for IGP Firing Rate
//' 
//' Constructs credible intervals for the time-inhomgeneous mean parameter (I) of the
//' inverse Gaussian renewal process.
//' 
//' @name FR_CI_IGP
//' @param time Vector of time points at which pointwise credible intervals will be constructed
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param basis_coef_samp Matrix of MCMC samples of the basis coefficients (\code{basis_coef} from \code{Sampler_IGP})
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in
//' @param alpha Double indicating the size of the credible interval ((1 - alpha) * 100 percent)
//' 
//' @returns List containing:
//' \describe{
//'   \item{\code{FR_MCMC_Samps}}{Matrix of MCMC samples of the firing rate at the specified time points}
//'   \item{\code{FR_CI}}{Matrix containing the upper and lower values of the credible interval at the specified time points (first column is lower value, second column is upper value)}
//'   \item{\code{FR_Median}}{Vector containing the estimated posterior median at the specified time points}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{alpha}}{must be between 0 and 0.5}
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
//' 
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
Rcpp::List FR_CI_IGP(const arma::vec time,
                     const int basis_degree,
                     const arma::vec boundary_knots,
                     const arma::vec internal_knots,
                     const arma::mat basis_coef_samp,
                     const arma::mat theta,
                     const double burnin_prop = 0.3,
                     const double alpha = 0.05){
  arma::mat basis_funct;
  splines2::BSpline bspline;
  
  bspline = splines2::BSpline(time, internal_knots, basis_degree,
                              boundary_knots);
  arma::mat bspline_mat{bspline.basis(true)};
  basis_funct = bspline_mat;
  
  int n_MCMC = basis_coef_samp.n_rows;
  arma::mat MCMC_FR = arma::zeros(std::floor((1 - burnin_prop) * n_MCMC), time.n_elem);
  arma::mat FR_CI = arma::zeros(time.n_elem, 2);
  arma::vec FR_median = arma::zeros(time.n_elem);
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  for(int i = burnin_num; i < n_MCMC; i++){
    MCMC_FR.row(i - burnin_num) = theta(i,0) + (basis_funct * basis_coef_samp.row(i).t()).t();
  }
  
  arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
  arma::vec q = arma::zeros(3);
  for(int i = 0; i < time.n_elem; i++){
    q = arma::quantile(MCMC_FR.col(i), p);
    FR_CI(i,0) = q(0);
    FR_median(i) = q(1);
    FR_CI(i,1) = q(2);
  }
  
  Rcpp::List output =  Rcpp::List::create(Rcpp::Named("FR_MCMC_Samps", MCMC_FR),
                                          Rcpp::Named("FR_CI", FR_CI),
                                          Rcpp::Named("FR_median", FR_median));
  
  return output;
}

//[[Rcpp::export]]
Rcpp::List FR_CI_Competition(const arma::vec time,
                             const int basis_degree,
                             const arma::vec boundary_knots,
                             const arma::vec internal_knots,
                             const arma::mat basis_coef_A_samp,
                             const arma::mat basis_coef_B_samp,
                             const arma::mat theta,
                             const double burnin_prop = 0.3,
                             const double alpha = 0.05){
  arma::mat basis_funct;
  splines2::BSpline bspline;
  
  bspline = splines2::BSpline(time, internal_knots, basis_degree,
                              boundary_knots);
  arma::mat bspline_mat{bspline.basis(true)};
  basis_funct = bspline_mat;
  
  int n_MCMC = basis_coef_A_samp.n_rows;
  arma::mat MCMC_A_FR = arma::zeros(std::floor((1 - burnin_prop) * n_MCMC), time.n_elem);
  arma::mat MCMC_B_FR = arma::zeros(std::floor((1 - burnin_prop) * n_MCMC), time.n_elem);
  arma::mat A_FR_CI = arma::zeros(time.n_elem, 2);
  arma::mat B_FR_CI = arma::zeros(time.n_elem, 2);
  arma::vec A_FR_median = arma::zeros(time.n_elem);
  arma::vec B_FR_median = arma::zeros(time.n_elem);
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  for(int i = burnin_num; i < n_MCMC; i++){
    MCMC_A_FR.row(i - burnin_num) = theta(i,0) + (basis_funct * basis_coef_A_samp.row(i).t()).t();
    MCMC_B_FR.row(i - burnin_num) = theta(i,1) + (basis_funct * basis_coef_B_samp.row(i).t()).t();
  }
  
  arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
  arma::vec q = arma::zeros(3);
  for(int i = 0; i < time.n_elem; i++){
    q = arma::quantile(MCMC_A_FR.col(i), p);
    A_FR_CI(i,0) = q(0);
    A_FR_median(i) = q(1);
    A_FR_CI(i,1) = q(2);
    q = arma::quantile(MCMC_B_FR.col(i), p);
    B_FR_CI(i,0) = q(0);
    B_FR_median(i) = q(1);
    B_FR_CI(i,1) = q(2);
  }
  
  Rcpp::List output =  Rcpp::List::create(Rcpp::Named("A_FR_MCMC_Samps", MCMC_A_FR),
                                          Rcpp::Named("B_FR_MCMC_Samps", MCMC_B_FR),
                                          Rcpp::Named("A_FR_CI", A_FR_CI),
                                          Rcpp::Named("B_FR_CI", B_FR_CI),
                                          Rcpp::Named("A_FR_median", A_FR_median),
                                          Rcpp::Named("B_FR_median", B_FR_median));
  
  return output;
}
