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
//' @name GetBSpline
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
//' B <- GetBSpline(time, basis_degree, boundary_knots, internal_knots)
//' 
//' @export
//[[Rcpp::export]]
arma::mat GetBSpline(const arma::vec time,
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
  arma::mat bspline_mat{bspline.basis(false)};
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
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
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
//' @param nu Double containing the value for the degrees of freedom parameter of the folded-t prior distribution on I_A_sigma and I_B_sigma
//' @param gamma Double containing the value for the scale parameter of the folded-t prior distribution on I_A_sigma and I_B_sigma
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
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
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
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' end_time <- 1
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2,
//'                                time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2)
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
                               const double end_time,
                               const bool time_inhomogeneous = true,
                               int Warm_block1 = 500,
                               int Warm_block2 = 4000,
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
                               double nu = 5,
                               double gamma = 0.25,
                               int delta_adaptation_block = 100,
                               int Mass_adaptation_block = 500,
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
 if(gamma <= 0){
   Rcpp::stop("'gamma' must be greater than 0");
 }
 for(int i = 0; i < n_A.n_elem; i++){
   if(arma::accu(X_A(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See A condition spike train " + std::to_string(i + 1) + ")");
   }
 }
 for(int i = 0; i < n_B.n_elem; i++){
   if(arma::accu(X_B(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See B condition spike train " + std::to_string(i + 1) + ")");
   }
 }
 for(int i = 0; i < n_AB.n_elem; i++){
   if(arma::accu(X_AB(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See AB condition spike train " + std::to_string(i + 1) + ")");
   }
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
     arma::vec time = arma::zeros(n_A(i) + 1);
     for(int j = 1; j < (n_A(i) + 1); j++){
       time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_A(i,0) = bspline_mat;
   }
   
   // Make spline basis for B functions
   arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
   for(int i = 0; i < n_B.n_elem; i++){
     arma::vec time = arma::zeros(n_B(i) + 1);
     for(int j = 1; j < (n_B(i) + 1); j++){
       time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_B(i,0) = bspline_mat;
   }
   
   // Make spline basis for AB functions
   arma::field<arma::mat> basis_funct_AB(n_AB.n_elem,1);
   for(int i = 0; i < n_AB.n_elem; i++){
     arma::vec time = arma::zeros(n_AB(i) + 1);
     for(int j = 1; j < (n_AB(i) + 1); j++){
       time(j) = arma::accu(X_AB(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_AB(i,0) = bspline_mat;
   }
   
   param = NeuralComp::Mixed_sampler_int_TI(basis_funct_A, basis_funct_B, basis_funct_AB,
                                            X_A, X_B, X_AB, n_A, n_B, n_AB, end_time, MCMC_iters, 
                                            Leapfrog_steps, I_A_mean, I_A_shape,
                                            I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
                                            sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                            step_size_theta, step_size_FR,
                                            delta_proposal_mean, delta_proposal_sd, 
                                            alpha_labels, nu, gamma, delta_adaptation_block,
                                            Mass_adaptation_block, M_proposal, 
                                            Warm_block1, Warm_block2);
 }else{
   param = NeuralComp::Mixed_sampler_int(X_A, X_B, X_AB, n_A, n_B, n_AB, end_time, MCMC_iters, 
                                         Leapfrog_steps, I_A_mean, I_A_shape,
                                         I_B_mean, I_B_shape, sigma_A_mean, sigma_A_shape,
                                         sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                         step_size_theta, delta_proposal_mean, delta_proposal_sd, 
                                         alpha_labels, delta_adaptation_block,
                                         Mass_adaptation_block, M_proposal, 
                                         Warm_block1, Warm_block2);
 }
 
 return param;
}

//' Sampler for IIGPP model
//' 
//' Conducts MCMC to get posterior samples from an (inhomogeneous) inverse Gaussian point process.
//' This function can fit a time-homogeneous model, as well as a time-inhomogeneous 
//' model.
//' 
//' @name Sampler_IIGPP
//' @param X List of vectors containing the ISIs the trials
//' @param n Vector containing number of spikes for the trials
//' @param MCMC_iters Integer containing the number of MCMC_iterations excluding warm up blocks
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
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
//' @param nu Double containing the value for the degrees of freedom parameter of the folded-t prior distribution on I_A_sigma and I_B_sigma
//' @param gamma Double containing the value for the scale parameter of the folded-t prior distribution on I_A_sigma and I_B_sigma
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
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
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
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                          internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                          Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots, 
//'                          internal_knots, end_time, Warm_block1 = Warm_block1,
//'                          Warm_block2 = Warm_block2)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List Sampler_IIGPP(const arma::field<arma::vec> X,
                         const arma::vec n,
                         int MCMC_iters,
                         const int basis_degree,
                         const arma::vec boundary_knots,
                         const arma::vec internal_knots,
                         const double end_time,
                         const bool time_inhomogeneous = true,
                         int Warm_block1 = 500,
                         int Warm_block2 = 2000,
                         int Leapfrog_steps = 10,
                         const double I_mean = 40, 
                         const double I_shape = 1,
                         const double sigma_mean = 6.32,
                         const double sigma_shape = 1,
                         double step_size_theta =  0.001,
                         double step_size_FR =  0.001,
                         double nu = 5,
                         double gamma = 0.25,
                         int Mass_adaptation_block = 500,
                         int M_proposal = 10){
  Rcpp::List param;
  
  // Check conditions
  if(I_mean <= 0){
    Rcpp::stop("'I_mean' must be greater than 0");
  }
  if(I_shape <= 0){
    Rcpp::stop("'I_shape' must be greater than 0");
  }
  if(sigma_mean <= 0){
    Rcpp::stop("'sigma_mean' must be greater than 0");
  }
  if(sigma_shape <= 0){
    Rcpp::stop("'sigma_shape' must be greater than 0");
  }
  if(step_size_theta <= 0){
    Rcpp::stop("'step_size_theta' must be greater than 0");
  }
  if(step_size_FR <= 0){
    Rcpp::stop("'step_size_FR' must be greater than 0");
  }
  if(gamma <= 0){
    Rcpp::stop("'gamma' must be greater than 0");
  }
  for(int i = 0; i < n.n_elem; i++){
    if(arma::accu(X(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See spike train " + std::to_string(i + 1) + ")");
    }
  }
  
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
    arma::field<arma::mat> basis_funct(n.n_elem,1);
    for(int i = 0; i < n.n_elem; i++){
      arma::vec time = arma::zeros(n(i) + 1);
      for(int j = 1; j < (n(i) + 1); j++){
        time(j) = arma::accu(X(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct(i,0) = bspline_mat;
    }
    
    param = NeuralComp::Mixed_sampler_IGP_int_TI(basis_funct, X, n, end_time, MCMC_iters, 
                                                 Leapfrog_steps, I_mean, I_shape,
                                                 sigma_mean, sigma_shape,
                                                 step_size_theta, step_size_FR,
                                                 nu, gamma, Mass_adaptation_block, 
                                                 Warm_block1, Warm_block2);
  }else{
    param = NeuralComp::Mixed_sampler_IGP_int(X, n, end_time, MCMC_iters, Leapfrog_steps, I_mean,
                                              I_shape, sigma_mean, sigma_shape,
                                              step_size_theta, Mass_adaptation_block,
                                              Warm_block1, Warm_block2);
  }
  
  return param;
}

//' Constructs CI for IIGPP Firing Rate
//' 
//' Constructs credible intervals for the time-inhomgeneous input current function (I) of the inhomogeneous
//' inverse Gaussian point process (IIGPP) model.
//' 
//' @name FR_CI_IIGPP
//' @param time Vector of time points at which pointwise credible intervals will be constructed
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param Results List produced from running \code{Sampler_IIGPP}
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in
//' @param alpha Double indicating the size of the credible interval ((1 - alpha) * 100 percent)
//' 
//' @returns List containing:
//' \describe{
//'   \item{\code{FR_MCMC_Samps}}{Matrix of MCMC samples of the input current function at the specified time points}
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
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots, 
//'                          internal_knots, end_time, Warm_block1 = Warm_block1,
//'                          Warm_block2 = Warm_block2)
//' 
//' ## Get CI
//' time <- seq(0, 1, 0.01)
//' CI <- FR_CI_IIGPP(time, basis_degree, boundary_knots, internal_knots, results)
//' 
//' ## plot CI
//' CI_range <- c(min(CI$FR_CI)-5, max(CI$FR_CI) + 5)
//' plot(time, CI$FR_median, type = 'l', col = "red", ylim = CI_range)
//' lines(time, CI$FR_CI[,1], col = "blue")
//' lines(time, CI$FR_CI[,2], col = "blue")
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List FR_CI_IIGPP(const arma::vec time,
                       const int basis_degree,
                       const arma::vec boundary_knots,
                       const arma::vec internal_knots,
                       const Rcpp::List Results,
                       const double burnin_prop = 0.3,
                       const double alpha = 0.05){
  
  const arma::mat basis_coef_samp = Results["basis_coef"];
  const arma::mat theta = Results["theta"];
  
  // Check conditions
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be in [0,1)");
  }
  if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be in [0,1)");
  }
  if(alpha <= 0){
    Rcpp::stop("'alpha' must be in (0,1)");
  }
  if(alpha >= 1){
    Rcpp::stop("'alpha' must be in (0,1)");
  }
  
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
  arma::mat bspline_mat{bspline.basis(false)};
  basis_funct = bspline_mat;
  
  int n_MCMC = basis_coef_samp.n_rows;
  arma::mat MCMC_FR = arma::zeros(std::floor((1 - burnin_prop) * n_MCMC), time.n_elem);
  arma::mat FR_CI = arma::zeros(time.n_elem, 2);
  arma::vec FR_median = arma::zeros(time.n_elem);
  int burnin_num = n_MCMC - std::floor((1 - burnin_prop) * n_MCMC);
  for(int i = burnin_num; i < n_MCMC; i++){
    MCMC_FR.row(i - burnin_num) = theta(i,0) * arma::exp(basis_funct * basis_coef_samp.row(i).t()).t();
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

//' Constructs CI for Competition Firing Rate
//' 
//' Constructs credible intervals for the time-inhomgeneous input current (I) of the 
//' competition model for time-inhomogeneous models.
//' 
//' @name FR_CI_Competition
//' @param time Vector of time points at which pointwise credible intervals will be constructed
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param Results List produced from running \code{Sampler_Competition}
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in
//' @param alpha Double indicating the size of the credible interval ((1 - alpha) * 100 percent)
//' 
//' @returns List containing:
//' \describe{
//'   \item{\code{A_FR_MCMC_Samps}}{Matrix of MCMC samples of the A input current function at the specified time points}
//'   \item{\code{B_FR_MCMC_Samps}}{Matrix of MCMC samples of the B input current function at the specified time points}
//'   \item{\code{A_FR_CI}}{Matrix containing the upper and lower values of the A input current function credible interval at the specified time points (first column is lower value, second column is upper value)}
//'   \item{\code{B_FR_CI}}{Matrix containing the upper and lower values of the B input current function credible interval at the specified time points (first column is lower value, second column is upper value)}
//'   \item{\code{A_FR_Median}}{Vector containing the estimated posterior median A input current function at the specified time points}
//'   \item{\code{B_FR_Median}}{Vector containing the estimated posterior median B input current function at the specified time points}
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
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2)
//' 
//' ## Get CI
//' time <- seq(0, 1, 0.01)
//' CI <- FR_CI_Competition(time, basis_degree, boundary_knots, internal_knots, results)
//' 
//' ## plot A input current CI
//' CI_range <- c(min(CI$A_FR_CI)-5, max(CI$A_FR_CI) + 5)
//' plot(time, CI$A_FR_median, type = 'l', col = "red", ylim = CI_range)
//' lines(time, CI$A_FR_CI[,1], col = "blue")
//' lines(time, CI$A_FR_CI[,2], col = "blue")
//' 
//' ## plot B input current CI
//' CI_range <- c(min(CI$B_FR_CI)-5, max(CI$B_FR_CI) + 5)
//' plot(time, CI$B_FR_median, type = 'l', col = "red", ylim = CI_range)
//' lines(time, CI$B_FR_CI[,1], col = "blue")
//' lines(time, CI$B_FR_CI[,2], col = "blue")
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List FR_CI_Competition(const arma::vec time,
                             const int basis_degree,
                             const arma::vec boundary_knots,
                             const arma::vec internal_knots,
                             const Rcpp::List Results,
                             const double burnin_prop = 0.3,
                             const double alpha = 0.05){
  const arma::mat basis_coef_A_samp = Results["basis_coef_A"];
  const arma::mat basis_coef_B_samp = Results["basis_coef_B"];
  const arma::mat theta = Results["theta"];
  
  // Check conditions
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be in [0,1)");
  }
  if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be in [0,1)");
  }
  if(alpha <= 0){
    Rcpp::stop("'alpha' must be in (0,1)");
  }
  if(alpha >= 1){
    Rcpp::stop("'alpha' must be in (0,1)");
  }
  
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
  arma::mat bspline_mat{bspline.basis(false)};
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
    MCMC_A_FR.row(i - burnin_num) = theta(i,0) * arma::exp(basis_funct * basis_coef_A_samp.row(i).t()).t();
    MCMC_B_FR.row(i - burnin_num) = theta(i,1) * arma::exp(basis_funct * basis_coef_B_samp.row(i).t()).t();
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


//' Calculates WAIC for the Competition Model
//' 
//' This function calculates the Watanabe-Akaike information criterion (WAIC) for 
//' the drift-diffusion competition model using the marginal likelihood (marginalizing out the labels).
//' This function will use the output from 
//' \code{Sampler_Competition}. The WAIC is defined on the deviance scale as waic = -2(lppd - p),
//' where lppd is the log pointwise predictive density, and p is the effective number of parameters.
//' The Marginal WAIC is akin to leave-one-spike-train-out cross validation (asymptotically).
//' 
//' @name WAIC_Competition
//' @param X_A List of vectors containing the ISIs of A trials
//' @param X_B List of vectors containing the ISIs of B trials
//' @param X_AB List of vectors containing the ISIs of AB trials
//' @param n_A Vector containing number of spikes for each A trial
//' @param n_B Vector containing number of spikes for each B trial
//' @param n_AB Vector containing number of spikes for each AB trial
//' @param Results List produced from running \code{Sampler_Competition}
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @returns List containing:
//' \describe{
//'   \item{\code{WAIC}}{Estimate of WAIC}
//'   \item{\code{LPPD}}{Estimate of LPPD}
//'   \item{\code{Effective_pars}}{Estimated Effective number of parameters}
//'   \item{\code{llik_A}}{Log-likelihood for A trial spike trains}
//'   \item{\code{llik_B}}{Log-likelihood for B trial spike trains}
//'   \item{\code{llik_AB}}{Marginal log-likelihood for AB trial spike trains}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2,
//'                                time_inhomogeneous = FALSE)
//'                                
//' ## Calculate WAIC
//' WAIC <- WAIC_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB,
//'                          results, basis_degree, boundary_knots, internal_knots, end_time,
//'                          time_inhomogeneous = FALSE)
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2)
//' 
//' ## Calculate WAIC
//' WAIC <- WAIC_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB,
//'                          results, basis_degree, boundary_knots, internal_knots,
//'                          end_time)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List WAIC_Competition(const arma::field<arma::vec> X_A,
                            const arma::field<arma::vec> X_B,
                            const arma::field<arma::vec> X_AB,
                            const arma::vec n_A,
                            const arma::vec n_B,
                            const arma::vec n_AB,
                            Rcpp::List Results,
                            const int basis_degree,
                            const arma::vec boundary_knots,
                            const arma::vec internal_knots,
                            const double end_time,
                            const bool time_inhomogeneous = true,
                            const double burnin_prop = 0.2){
  // Check if max_time is large enough
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  
  for(int i = 0; i < n_A.n_elem; i++){
    if(arma::accu(X_A(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See A condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    if(arma::accu(X_B(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See B condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  for(int i = 0; i < n_AB.n_elem; i++){
    if(arma::accu(X_AB(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See AB condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  arma::field<arma::mat> basis_funct_A(n_A.n_elem,1);
  arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
  arma::field<arma::mat> basis_funct_AB(n_AB.n_elem,1);
  arma::mat basis_coef_A;
  arma::mat basis_coef_B;
  arma::mat theta = Results["theta"];
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
    for(int i = 0; i < n_A.n_elem; i++){
      arma::vec time = arma::zeros(n_A(i) + 1);
      for(int j = 1; j < (n_A(i) + 1); j++){
        time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_A(i,0) = bspline_mat;
    }
    
    for(int i = 0; i < n_B.n_elem; i++){
      arma::vec time = arma::zeros(n_B(i) + 1);
      for(int j = 1; j < (n_B(i) + 1); j++){
        time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_B(i,0) = bspline_mat;
    }
    
    for(int i = 0; i < n_AB.n_elem; i++){
      arma::vec time = arma::zeros(n_AB(i) + 1);
      for(int j = 1; j < (n_AB(i) + 1); j++){
        time(j) = arma::accu(X_AB(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_AB(i,0) = bspline_mat;
    }
    arma::mat ph = Results["basis_coef_A"];
    basis_coef_A = ph;
    arma::mat ph1 = Results["basis_coef_B"];
    basis_coef_B = ph1;
  }else{
    for(int i = 0; i < n_A.n_elem; i++){
      basis_funct_A(i, 0) = arma::zeros(n_A(i) + 1, 1);
    }
    for(int i = 0; i < n_B.n_elem; i++){
      basis_funct_B(i, 0) = arma::zeros(n_B(i) + 1, 1);
    }
    for(int i = 0; i < n_AB.n_elem; i++){
      basis_funct_AB(i, 0) = arma::zeros(n_AB(i) + 1, 1);
    }
    basis_coef_A = arma::zeros(theta.n_rows,1);
    basis_coef_B = arma::zeros(theta.n_rows,1);
    
  }
  Rcpp::List waic;
  waic = NeuralComp::calc_WAIC_competition_Marginal(X_A, X_B, X_AB, n_A, n_B, n_AB, end_time, theta, basis_coef_A,
                                                    basis_coef_B, basis_funct_A, basis_funct_B, basis_funct_AB,
                                                    burnin_prop);

  
  return waic;
}


//' Calculates WAIC for the IIGPP Model
//' 
//' This function calculates the Watanabe-Akaike information criterion (WAIC) for 
//' the (inhomogeneous) inverse Gaussian point process (IIGPP) model. This function will use the output from
//' \code{Sampler_IIGPP} fit for the A, B, and AB data. The WAIC is defined on the 
//' deviance scale as waic = -2(lppd - p), where lppd is the log pointwise 
//' predictive density, and p is the effective number of parameters.
//' 
//' @name WAIC_IIGPP
//' @param X_A List of vectors containing the ISIs of A trials
//' @param X_B List of vectors containing the ISIs of B trials
//' @param X_AB List of vectors containing the ISIs of AB trials
//' @param n_A Vector containing number of spikes for each A trial
//' @param n_B Vector containing number of spikes for each B trial
//' @param n_AB Vector containing number of spikes for each AB trial
//' @param Results_A List produced from running \code{Sampler_IIGPP} for A trials
//' @param Results_B List produced from running \code{Sampler_IIGPP} for B trials
//' @param Results_AB List produced from running \code{Sampler_IIGPP} for AB trials
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-spline
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @returns List containing:
//' \describe{
//'   \item{\code{WAIC}}{Estimate of WAIC}
//'   \item{\code{LPPD}}{Estimate of LPPD}
//'   \item{\code{Effective_pars}}{Estimated Effective number of parameters}
//'   \item{\code{llik_A}}{Log-likelihood for A trial spike trains}
//'   \item{\code{llik_B}}{Log-likelihood for B trial spike trains}
//'   \item{\code{llik_AB}}{Log-likelihood for AB trial spike trains}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain for A trials
//' results_A <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1,
//'                            Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'                        
//' ## Run MCMC chain for B trials
//' results_B<- Sampler_IIGPP(dat$X_B, dat$n_B, MCMC_iters, basis_degree, boundary_knots,
//'                           internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                           Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'
//' ## Run MCMC chain for AB trials
//' results_AB <- Sampler_IIGPP(dat$X_AB, dat$n_AB, MCMC_iters, basis_degree, boundary_knots,
//'                             internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                             Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)         
//' ## Calculate WAIC
//' WAIC <- WAIC_IIGPP(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, results_A,
//'                    results_B, results_AB, basis_degree, boundary_knots, internal_knots,
//'                    end_time, time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain for A trials
//' results_A <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                            Warm_block2 = Warm_block2)
//'                        
//' ## Run MCMC chain for B trials
//' results_B<- Sampler_IIGPP(dat$X_B, dat$n_B, MCMC_iters, basis_degree, boundary_knots,
//'                           internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                           Warm_block2 = Warm_block2)
//'
//' ## Run MCMC chain for AB trials
//' results_AB <- Sampler_IIGPP(dat$X_AB, dat$n_AB, MCMC_iters, basis_degree, boundary_knots,
//'                             internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                             Warm_block2 = Warm_block2)
//'
//' WAIC <- WAIC_IIGPP(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, results_A,
//'                    results_B, results_AB, basis_degree, boundary_knots, internal_knots,
//'                    end_time)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List WAIC_IIGPP(const arma::field<arma::vec> X_A,
                      const arma::field<arma::vec> X_B,
                      const arma::field<arma::vec> X_AB,
                      const arma::vec n_A,
                      const arma::vec n_B,
                      const arma::vec n_AB,
                      Rcpp::List Results_A,
                      Rcpp::List Results_B,
                      Rcpp::List Results_AB,
                      const int basis_degree,
                      const arma::vec boundary_knots,
                      const arma::vec internal_knots,
                      const double end_time,
                      const bool time_inhomogeneous = true,
                      const double burnin_prop = 0.2){
 
 if(burnin_prop < 0){
   Rcpp::stop("'burnin_prop' must be between 0 and 1");
 }if(burnin_prop >= 1){
   Rcpp::stop("'burnin_prop' must be between 0 and 1");
 }
 for(int i = 0; i < n_A.n_elem; i++){
   if(arma::accu(X_A(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See A condition spike train " + std::to_string(i + 1) + ")");
   }
 }
 for(int i = 0; i < n_B.n_elem; i++){
   if(arma::accu(X_B(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See B condition spike train " + std::to_string(i + 1) + ")");
   }
 }
 for(int i = 0; i < n_AB.n_elem; i++){
   if(arma::accu(X_AB(i,0)) > end_time + arma::datum::eps){
     Rcpp::stop("All spikes must be observed in [0,'end_time'] (See AB condition spike train " + std::to_string(i + 1) + ")");
   }
 }
 arma::field<arma::mat> basis_funct_A(n_A.n_elem,1);
 arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
 arma::field<arma::mat> basis_funct_AB(n_AB.n_elem,1);
 arma::mat basis_coef_A;
 arma::mat basis_coef_B;
 arma::mat basis_coef_AB;
 arma::mat theta_A = Results_A["theta"];
 arma::mat theta_B = Results_B["theta"];
 arma::mat theta_AB = Results_AB["theta"];
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
   for(int i = 0; i < n_A.n_elem; i++){
     arma::vec time = arma::zeros(n_A(i) + 1);
     for(int j = 1; j < (n_A(i) + 1); j++){
       time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_A(i,0) = bspline_mat;
   }
   
   for(int i = 0; i < n_B.n_elem; i++){
     arma::vec time = arma::zeros(n_B(i) + 1);
     for(int j = 1; j < (n_B(i) + 1); j++){
       time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_B(i,0) = bspline_mat;
   }
   
   for(int i = 0; i < n_AB.n_elem; i++){
     arma::vec time = arma::zeros(n_AB(i) + 1);
     for(int j = 1; j < (n_AB(i) + 1); j++){
       time(j) = arma::accu(X_AB(i,0).subvec(0,j-1));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(false)};
     basis_funct_AB(i,0) = bspline_mat;
   }
   arma::mat ph = Results_A["basis_coef"];
   basis_coef_A = ph;
   arma::mat ph1 = Results_B["basis_coef"];
   basis_coef_B = ph1;
   arma::mat ph2 = Results_AB["basis_coef"];
   basis_coef_AB = ph2;
 }else{
   for(int i = 0; i < n_A.n_elem; i++){
     basis_funct_A(i, 0) = arma::zeros(n_A(i) + 1, 1);
   }
   for(int i = 0; i < n_B.n_elem; i++){
     basis_funct_B(i, 0) = arma::zeros(n_B(i) + 1, 1);
   }
   for(int i = 0; i < n_AB.n_elem; i++){
     basis_funct_AB(i, 0) = arma::zeros(n_AB(i) + 1, 1);
   }
   basis_coef_A = arma::zeros(theta_A.n_rows,1);
   basis_coef_B = arma::zeros(theta_B.n_rows,1);
   basis_coef_AB = arma::zeros(theta_AB.n_rows,1);
 }
 Rcpp::List waic = NeuralComp::calc_WAIC_IGP_observation(X_A, X_B, X_AB, n_A, n_B, n_AB, end_time, theta_A, 
                                                         basis_coef_A, theta_B, basis_coef_B, 
                                                         theta_AB, basis_coef_AB, basis_funct_A, 
                                                         basis_funct_B, basis_funct_AB, burnin_prop);
 

 return waic;
}

//' Calculates WAIC for the Winner-take-all Model
//' 
//' This function calculates the Watanabe-Akaike information criterion (WAIC) for 
//' the winner-take-all model, which can be constructed via IIGPP models. This 
//' function will use the output from \code{Sampler_IIGPP} fit for the A, B, and 
//' AB data. The WAIC is defined on the deviance scale as waic = -2(lppd - p), 
//' where lppd is the log pointwise predictive density, and p is the effective 
//' number of parameters.
//' 
//' @name WAIC_Winner_Take_All
//' @param X_A List of vectors containing the ISIs of A trials (and AB trials if testing winner-take-all (A stimulus))
//' @param X_B List of vectors containing the ISIs of B trials (and AB trials if testing winner-take-all (B stimulus))
//' @param n_A Vector containing number of spikes for each A trial (and AB trial if testing winner-take-all (A stimulus))
//' @param n_B Vector containing number of spikes for each B trial (and AB trial if testing winner-take-all (B stimulus))
//' @param Results_A List produced from running \code{Sampler_IIGPP} for A trials (and AB trials if testing winner-take-all (A stimulus))
//' @param Results_B List produced from running \code{Sampler_IIGPP} for B trials (and AB trials if testing winner-take-all (B stimulus))
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @returns List containing:
//' \describe{
//'   \item{\code{WAIC}}{Estimate of WAIC}
//'   \item{\code{LPPD}}{Estimate of LPPD}
//'   \item{\code{Effective_pars}}{Estimated Effective number of parameters}
//'   \item{\code{llik_A}}{Log-likelihood for A trial spike trains (and AB trials if testing winner-take-all (A stimulus))}
//'   \item{\code{llik_B}}{Log-likelihood for B trial spike trains (and AB trials if testing winner-take-all (B stimulus))}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ################################################
//' ## Test Winner-take-all (Encoding Stimulus A) ##
//' ################################################
//' 
//' ### Create new dataset of A and AB trials together
//' dat_A_AB <- c(dat$X_A, dat$X_AB)
//' n_A_AB <- c(dat$n_A, dat$n_AB)
//' 
//' ## Run MCMC chain for A trials and AB trials
//' results_A_AB <- Sampler_IIGPP(dat_A_AB, n_A_AB, MCMC_iters, basis_degree, boundary_knots,
//'                               internal_knots, end_time, Warm_block1 = Warm_block1,
//'                               Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'                        
//' ## Run MCMC chain for B trials
//' results_B <- Sampler_IIGPP(dat$X_B, dat$n_B, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                            Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'
//' ## Calculate WAIC - Encoding stimulus A
//' WAIC <- WAIC_Winner_Take_All(dat_A_AB, dat$X_B, n_A_AB, dat$n_B, results_A_AB,
//'                              results_B, basis_degree, boundary_knots,
//'                              internal_knots, end_time, time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ################################################
//' ## Test Winner-take-all (Encoding Stimulus B) ##
//' ################################################
//' 
//' ### Create new dataset of A and AB trials together
//' dat_B_AB <- c(dat$X_B, dat$X_AB)
//' n_B_AB <- c(dat$n_B, dat$n_AB)
//' 
//' ## Run MCMC chain for A trials 
//' results_A <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1,
//'                            Warm_block2 = Warm_block2)
//'                        
//' ## Run MCMC chain for B trials and AB trials
//' results_B_AB <- Sampler_IIGPP(dat_B_AB, n_B_AB, MCMC_iters, basis_degree, boundary_knots,
//'                               internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                               Warm_block2 = Warm_block2)
//'
//' ## Calculate WAIC - Encoding stimulus B
//' WAIC <- WAIC_Winner_Take_All(dat$X_A, dat_B_AB, dat$n_A, n_B_AB, results_A,
//'                              results_B_AB, basis_degree, boundary_knots,
//'                              internal_knots, end_time)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List WAIC_Winner_Take_All(const arma::field<arma::vec> X_A,
                                const arma::field<arma::vec> X_B,
                                const arma::vec n_A,
                                const arma::vec n_B,
                                Rcpp::List Results_A,
                                Rcpp::List Results_B,
                                const int basis_degree,
                                const arma::vec boundary_knots,
                                const arma::vec internal_knots,
                                const double end_time, 
                                const bool time_inhomogeneous = true,
                                const double burnin_prop = 0.2){
  
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  for(int i = 0; i < n_A.n_elem; i++){
    if(arma::accu(X_A(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See A condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    if(arma::accu(X_B(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See B condition spike train " + std::to_string(i + 1) + ")");
    }
  }

  arma::field<arma::mat> basis_funct_A(n_A.n_elem,1);
  arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
  arma::mat basis_coef_A;
  arma::mat basis_coef_B;
  arma::mat theta_A = Results_A["theta"];
  arma::mat theta_B = Results_B["theta"];
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
    for(int i = 0; i < n_A.n_elem; i++){
      arma::vec time = arma::zeros(n_A(i) + 1);
      for(int j = 1; j < (n_A(i) + 1); j++){
        time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_A(i,0) = bspline_mat;
    }
    
    for(int i = 0; i < n_B.n_elem; i++){
      arma::vec time = arma::zeros(n_B(i) + 1);
      for(int j = 1; j < (n_B(i) + 1); j++){
        time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_B(i,0) = bspline_mat;
    }
    
    arma::mat ph = Results_A["basis_coef"];
    basis_coef_A = ph;
    arma::mat ph1 = Results_B["basis_coef"];
    basis_coef_B = ph1;
  }else{
    for(int i = 0; i < n_A.n_elem; i++){
      basis_funct_A(i, 0) = arma::zeros(n_A(i) + 1, 1);
    }
    for(int i = 0; i < n_B.n_elem; i++){
      basis_funct_B(i, 0) = arma::zeros(n_B(i) + 1, 1);
    }
    basis_coef_A = arma::zeros(theta_A.n_rows,1);
    basis_coef_B = arma::zeros(theta_B.n_rows,1);
  }
  Rcpp::List waic = NeuralComp::calc_WAIC_IGP_WTA(X_A, X_B, n_A, n_B, end_time, theta_A, 
                                                  basis_coef_A, theta_B, basis_coef_B,
                                                  basis_funct_A, basis_funct_B, burnin_prop);
  
  
  return waic;
}


//' Posterior Predictive Sampling for Competition Model
//' 
//' This function generates posterior predictive samples of scientific interest from
//' the competition model. Specifically, this function allows you to obtain (1) posterior
//' predictive samples from the A condition, B condition, and AB conditions (2) posterior
//' predictive samples of spike counts under the A condition, B condition, and AB conditions, 
//' (3) posterior predictive samples of time spent encoding each state (4) posterior predictive 
//' samples for the number of switches in a trial. This function is to be used after running
//' \code{Sampler_Competition}.
//' 
//' @name Competition_Posterior_Predictive
//' @param end_time Double containing length of trial to simulate
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param Results List produced from running \code{Sampler_Competition}
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param n_samples Integer containing number of posterior predictive samples to generate
//' @returns List containing:
//' \describe{
//'   \item{\code{posterior_pred_samples_A}}{Posterior predictive samples of spike trains under the A stimulus}
//'   \item{\code{posterior_pred_samples_B}}{Posterior predictive samples of spike trains under the B stimulus}
//'   \item{\code{posterior_pred_samples_AB}}{Posterior predictive samples of spike trains under the A  and B stimuli}
//'   \item{\code{posterior_pred_labels}}{labels corresponding to posterior_pred_samples_AB}
//'   \item{\code{n_A}}{Number of spikes in each of the spike trains in posterior_pred_samples_A}
//'   \item{\code{n_B}}{Number of spikes in each of the spike trains in posterior_pred_samples_B}
//'   \item{\code{n_AB}}{Number of spikes in each of the spike trains in posterior_pred_samples_AB}
//'   \item{\code{switch_times}}{Times spent in each encoding state in the spike trains generated in posterior_pred_samples_AB}
//'   \item{\code{switch_states}}{Encoding state (labels) corresponding to switch_times}
//'   \item{\code{n_switches}}{number of switches observed in each spike train generated in posterior_pred_samples_AB}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{n_sample}}{must be greater than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, 
//'                                Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'                                
//' ## Posterior Predictive Samples
//' post_pred <- Competition_Posterior_Predictive(end_time, basis_degree, boundary_knots, 
//'                                               internal_knots, results, 
//'                                               time_inhomogeneous = FALSE)
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_Competition(dat$X_A, dat$X_B, dat$X_AB, dat$n_A, dat$n_B, dat$n_AB, 
//'                                MCMC_iters, basis_degree, boundary_knots, internal_knots,
//'                                end_time, Warm_block1 = Warm_block1, Warm_block2 = Warm_block2)
//'                                
//' ## Posterior Predictive Samples                               
//' post_pred <- Competition_Posterior_Predictive(end_time, basis_degree, boundary_knots,
//'                                               internal_knots, results)
//'                                               
//' @export
//[[Rcpp::export]]
Rcpp::List Competition_Posterior_Predictive(const double end_time,
                                            const int basis_degree,
                                            const arma::vec boundary_knots,
                                            const arma::vec internal_knots,
                                            Rcpp::List Results,
                                            const double burnin_prop = 0.2,
                                            const bool time_inhomogeneous = true,
                                            const int n_samples = 10000){
  // check parameters
  if(end_time <= 0){
    Rcpp::stop("'end_time' must be greater than 0");
  }
  if(n_samples <= 0){
    Rcpp::stop("'n_samples' must be an integer greater or equal to 1");
  }
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  arma::mat basis_coef_A;
  arma::mat basis_coef_B;
  arma::mat theta = Results["theta"];
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
    arma::mat ph = Results["basis_coef_A"];
    basis_coef_A = ph;
    arma::mat ph1 = Results["basis_coef_B"];
    basis_coef_B = ph1;
  }else{
    basis_coef_A = arma::zeros(theta.n_rows,1);
    basis_coef_B = arma::zeros(theta.n_rows,1);
  }
  if(n_samples < 1){
    Rcpp::stop("'n_samples' must be an integer greater than 1");
  }
  
  Rcpp::List output = NeuralComp::posterior_pred_samples(theta, basis_coef_A, basis_coef_B,
                                                         basis_degree, boundary_knots, internal_knots,
                                                         burnin_prop, end_time, time_inhomogeneous, n_samples);

  return output;
}



//' Calculate Difference in LPPD to Ensure that the A and B Stimuli are Sufficiently Different
//' 
//' This function calculates the difference in LPPD between separate models for 
//' the A and B spike trains and a joint model for the A and B spike trains. The
//' key idea is that the separate models should be able to model the spike trains
//' significantly better than the joint model, indicating the two spike train 
//' distributions are significantly different between the A and B stimuli. The 
//' function will use the output from \code{Sampler_IIGPP} fit for the A, B, and 
//' joint data (A and B single-stimulus spike trains). 
//' 
//' @name Diff_LPPD
//' @param X_A List of vectors containing the ISIs of A trials
//' @param X_B List of vectors containing the ISIs of B trials
//' @param n_A Vector containing number of spikes for each A trial
//' @param n_B Vector containing number of spikes for each B trial
//' @param Results_A List produced from running \code{Sampler_IIGPP} for A trials
//' @param Results_B List produced from running \code{Sampler_IIGPP} for B trials
//' @param Results_joint List produced from running \code{Sampler_IIGPP} for the A and B trials combined
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @returns Double containing the difference in LPPD between the separate models and the joint model
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' 
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ### Fit Separate Models
//' 
//' ## Run MCMC chain for A trials
//' results_A <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1,
//'                            Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'                        
//' ## Run MCMC chain for B trials
//' results_B<- Sampler_IIGPP(dat$X_B, dat$n_B, MCMC_iters, basis_degree, boundary_knots,
//'                           internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                           Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//' 
//' ### Fit Joint Model
//' 
//' ### Create new dataset of A and B trials together
//' dat_A_B <- c(dat$X_A, dat$X_B)
//' n_A_B <- c(dat$n_A, dat$n_B)
//' 
//' ## Run MCMC chain for A trials and B trials jointly
//' results_A_B <- Sampler_IIGPP(dat_A_B, n_A_B, MCMC_iters, basis_degree, boundary_knots,
//'                              internal_knots, end_time, Warm_block1 = Warm_block1,
//'                              Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'                       
//'
//' ## Calculate Difference in LPPD
//' diff <- Diff_LPPD(dat$X_A, dat$X_B, dat$n_A, dat$n_B, results_A,
//'                   results_B, results_A_B, basis_degree, boundary_knots,
//'                   internal_knots, end_time, time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ### Fit Separate Models
//' 
//' ## Run MCMC chain for A trials
//' results_A <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots,
//'                            internal_knots, end_time, Warm_block1 = Warm_block1,
//'                            Warm_block2 = Warm_block2)
//'                        
//' ## Run MCMC chain for B trials
//' results_B<- Sampler_IIGPP(dat$X_B, dat$n_B, MCMC_iters, basis_degree, boundary_knots,
//'                           internal_knots, end_time, Warm_block1 = Warm_block1, 
//'                           Warm_block2 = Warm_block2)
//' 
//' ### Fit Joint Model
//' 
//' ### Create new dataset of A and B trials together
//' dat_A_B <- c(dat$X_A, dat$X_B)
//' n_A_B <- c(dat$n_A, dat$n_B)
//' 
//' ## Run MCMC chain for A trials and B trials jointly
//' results_A_B <- Sampler_IIGPP(dat_A_B, n_A_B, MCMC_iters, basis_degree, boundary_knots,
//'                              internal_knots, end_time, Warm_block1 = Warm_block1,
//'                              Warm_block2 = Warm_block2)
//'                       
//'
//' ## Calculate Difference in LPPD
//' diff <- Diff_LPPD(dat$X_A, dat$X_B, dat$n_A, dat$n_B, results_A,
//'                   results_B, results_A_B, basis_degree, boundary_knots,
//'                   internal_knots, end_time)
//' 
//' @export
//[[Rcpp::export]]
double Diff_LPPD(const arma::field<arma::vec> X_A,
                 const arma::field<arma::vec> X_B,
                 const arma::vec n_A,
                 const arma::vec n_B,
                 Rcpp::List Results_A,
                 Rcpp::List Results_B,
                 Rcpp::List Results_joint,
                 const int basis_degree,
                 const arma::vec boundary_knots,
                 const arma::vec internal_knots,
                 const double end_time,
                 const bool time_inhomogeneous = true,
                 const double burnin_prop = 0.2){
  for(int i = 0; i < n_A.n_elem; i++){
    if(arma::accu(X_A(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See A condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  for(int i = 0; i < n_B.n_elem; i++){
    if(arma::accu(X_B(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See B condition spike train " + std::to_string(i + 1) + ")");
    }
  }
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  arma::vec n_joint = arma::join_cols(n_A, n_B);
  arma::field<arma::vec> X_joint(n_joint.n_elem, 1);
  for(int i = 0; i < n_joint.n_elem; i++){
    if(i < n_A.n_elem){
      X_joint(i,0) = X_A(i,0);
    }else{
      X_joint(i,0) = X_B(i - n_A.n_elem, 0);
    }
  }
  arma::field<arma::mat> basis_funct_A(n_A.n_elem,1);
  arma::field<arma::mat> basis_funct_B(n_B.n_elem,1);
  arma::field<arma::mat> basis_funct_joint(n_joint.n_elem,1);
  arma::mat basis_coef_A;
  arma::mat basis_coef_B;
  arma::mat basis_coef_joint;
  arma::mat theta_A = Results_A["theta"];
  arma::mat theta_B = Results_B["theta"];
  arma::mat theta_joint = Results_joint["theta"];
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
    for(int i = 0; i < n_A.n_elem; i++){
      arma::vec time = arma::zeros(n_A(i) + 1);
      for(int j = 1; j < (n_A(i) + 1); j++){
        time(j) = arma::accu(X_A(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_A(i,0) = bspline_mat;
      basis_funct_joint(i,0) = bspline_mat;
    }
    
    for(int i = 0; i < n_B.n_elem; i++){
      arma::vec time = arma::zeros(n_B(i) + 1);
      for(int j = 1; j < (n_B(i) + 1); j++){
        time(j) = arma::accu(X_B(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct_B(i,0) = bspline_mat;
      basis_funct_joint(i + n_A.n_elem,0) = bspline_mat;
    }
    arma::mat ph = Results_A["basis_coef"];
    basis_coef_A = ph;
    arma::mat ph1 = Results_B["basis_coef"];
    basis_coef_B = ph1;
    arma::mat ph2 = Results_joint["basis_coef"];
    basis_coef_joint = ph2;
  }else{
    for(int i = 0; i < n_A.n_elem; i++){
      basis_funct_A(i, 0) = arma::zeros(n_A(i) + 1, 1);
    }
    for(int i = 0; i < n_B.n_elem; i++){
      basis_funct_B(i, 0) = arma::zeros(n_B(i) + 1, 1);
    }
    for(int i = 0; i < n_joint.n_elem; i++){
      basis_funct_joint(i, 0) = arma::zeros(n_joint(i) + 1, 1);
    }
    basis_coef_A = arma::zeros(theta_A.n_rows,1);
    basis_coef_B = arma::zeros(theta_B.n_rows,1);
    basis_coef_joint = arma::zeros(theta_joint.n_rows,1);
  }
  
  double ratio = NeuralComp::calc_Diff_LPPD_A_B(X_A, X_B, X_joint, n_A, n_B, n_joint, end_time, theta_A, 
                                                basis_coef_A, theta_B, basis_coef_B, 
                                                theta_joint, basis_coef_joint, basis_funct_A, 
                                                basis_funct_B, basis_funct_joint, burnin_prop);
  return ratio;
}


//' Calculates Posterior P-values for IIGPP Models
//' 
//' This function calculates the posterior p-value using different discrepency variables.
//' Specifically, this function uses the the following discrepency variables: (1) Average 
//' Log-Likelihood, (2) Mean of the Spike Counts, and (3) Variance of the Spike Counts. Using these
//' three discrepancy variables, this function calculates posterior p-values which
//' calculate the discrepency between the fitted model and the observed spike trains.
//' 
//' @name Test_IIGPP_Fit
//' @param X List of vectors containing the ISIs of one of the two (A or B) single stimulus trials
//' @param n Vector containing number of spikes for each one spike train corresponding to X
//' @param Results List produced from running \code{Sampler_IIGPP} for one of the two (A or B) single stimulus trials
//' @param basis_degree Integer indicating the degree of B-splines (3 for cubic splines)
//' @param boundary_knots Vector of two elements specifying the boundary knots
//' @param internal_knots Vector containing the desired internal knots of the B-splines
//' @param end_time Double containing the end of the experimental window (experimental window assumed [0, end_time])
//' @param time_inhomogeneous Boolean containing whether or not a time-inhomogeneous model should be used (if false then basis_degree, boundary_knots, and internal_knots can take any value of the correct type)
//' @param burnin_prop Double containing proportion of MCMC samples that should be discarded due to MCMC burn-in (Note burnin_prop includes warm-up iterations)
//' @returns List containing:
//' \describe{
//'   \item{\code{p_val}}{Posterior predictive p-value using average log-likelihood as the discrepency variable}
//'   \item{\code{p_val_mean_SC}}{Posterior predictive p-value using the mean of the spike counts as the discrepency variable}
//'   \item{\code{p_val_var_SC}}{Posterior predictive p-value using the variance of the spike counts as the discrepency variable}
//' }
//' 
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{burnin_prop}}{must be greater than or equal to 0 and less than 1}
//'   \item{\code{end_time}}{must be larger than 0 and all spike times must be before end_time}
//' }
//' 
//' @examples
//' ##############################
//' ### Time-Homogeneous Model ###
//' ##############################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_homogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots, 
//'                          internal_knots, end_time, Warm_block1 = Warm_block1,
//'                          Warm_block2 = Warm_block2, time_inhomogeneous = FALSE)
//'
//' posterior_p_val <- Test_IIGPP_Fit(dat$X_A, dat$n_A, results, basis_degree,
//'                                   boundary_knots, internal_knots, end_time,
//'                                   time_inhomogeneous = FALSE)
//' 
//' 
//' ################################
//' ### Time-Inhomogeneous Model ###
//' ################################
//' 
//' ## Load sample data
//' dat <- readRDS(system.file("test-data", "time_inhomogeneous_sample_dat.RDS",
//'                package = "NeuralComp"))
//' 
//' ## set parameters
//' MCMC_iters <- 100
//' basis_degree <- 3
//' boundary_knots <- c(0, 1)
//' internal_knots <- c(0.25, 0.5, 0.75)
//' end_time <- 1
//' 
//' ## Warm Blocks should be longer, however for the example, they are short
//' Warm_block1 = 50
//' Warm_block2 = 50
//' 
//' ## Run MCMC chain
//' results <- Sampler_IIGPP(dat$X_A, dat$n_A, MCMC_iters, basis_degree, boundary_knots, 
//'                          internal_knots, end_time, Warm_block1 = Warm_block1,
//'                          Warm_block2 = Warm_block2)
//'
//' posterior_p_val <- Test_IIGPP_Fit(dat$X_A, dat$n_A, results, basis_degree,
//'                                   boundary_knots, internal_knots, end_time)
//' 
//' @export
//[[Rcpp::export]]
Rcpp::List Test_IIGPP_Fit(const arma::field<arma::vec> X,
                          const arma::vec n,
                          Rcpp::List Results,
                          const int basis_degree,
                          const arma::vec boundary_knots,
                          const arma::vec internal_knots,
                          const double end_time,
                          const bool time_inhomogeneous = true,
                          const double burnin_prop = 0.2){
  arma::mat theta = Results["theta"];
  arma::field<arma::mat> basis_funct(n.n_elem,1);
  arma::mat basis_coef;
  for(int i = 0; i < n.n_elem; i++){
    if(arma::accu(X(i,0)) > end_time + arma::datum::eps){
      Rcpp::stop("All spikes must be observed in [0,'end_time'] (See spike train" + std::to_string(i + 1) + ")");
    }
  }
  
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
    for(int i = 0; i < n.n_elem; i++){
      arma::vec time = arma::zeros(n(i) + 1);
      for(int j = 1; j < (n(i) + 1); j++){
        time(j) = arma::accu(X(i,0).subvec(0,j-1));
      }
      bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix
      arma::mat bspline_mat{bspline.basis(false)};
      basis_funct(i,0) = bspline_mat;
    }
    
    arma::mat ph = Results["basis_coef"];
    basis_coef = ph;
  }else{
    for(int i = 0; i < n.n_elem; i++){
      basis_funct(i, 0) = arma::zeros(n(i) + 1, 1);
    }
    basis_coef = arma::zeros(theta.n_rows,1);
  }
  // Run bootstrap test
  Rcpp::List p_val = NeuralComp::calc_chi_squared_IIGPP(X, n, end_time, theta, basis_coef, basis_funct, end_time, basis_degree,
                                                        boundary_knots, internal_knots, time_inhomogeneous, burnin_prop);
  
  return p_val;
}

