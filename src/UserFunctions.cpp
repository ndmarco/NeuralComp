#include <RcppArmadillo.h>
#include <cmath>
#include <NeuralComp.h>
#include <splines2Armadillo.h>

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
              int Warm_block = 500,
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
              double step_size =  0.001,
              double step_size_delta =  0.00005,
              Rcpp::Nullable<Rcpp::NumericMatrix> Mass_mat = R_NilValue){
  arma::vec init_position1;
  if(init_position.isNull()){
    init_position1 = {50, 50, sqrt(50), sqrt(50), 0.001};
  }else{
    Rcpp::NumericVector X_(init_position);
    init_position1 = Rcpp::as<arma::vec>(X_);
  }
  arma::vec eps_step1;
  if(eps_step.isNull()){
    eps_step1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericVector X_(eps_step);
    eps_step1 = Rcpp::as<arma::vec>(X_);
  }
  arma::mat Mass_mat1;
  if(Mass_mat.isNull()){
    arma::vec diag_elem = {1, 1, 1, 1};
    Mass_mat1 = arma::diagmat(diag_elem);
  }else{
    Rcpp::NumericMatrix X_(Mass_mat);
    Mass_mat1 = Rcpp::as<arma::mat>(X_);
  }
  
  
  arma::mat theta = NeuralComp::HMC_sampler(Labels, X_A, X_B, X_AB, n_A, n_B, n_AB, init_position1,
                                            MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                            I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                            sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                            eps_step1, step_size, step_size_delta, Mass_mat1, 
                                            Warm_block);
   return theta;
}

//[[Rcpp::export]]
Rcpp::List FR_CI(const arma::vec time,
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
  int burnin_num = std::ceil(burnin_prop * n_MCMC);
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

//' HMC sampler for competition model
 //' 
 //' @name HMC_TI
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
Rcpp::List HMC_TI(arma::field<arma::vec> Labels,
                  const arma::field<arma::vec> X_A,
                  const arma::field<arma::vec> X_B,
                  const arma::field<arma::vec> X_AB,
                  const arma::vec n_A,
                  const arma::vec n_B,
                  const arma::vec n_AB,
                  int MCMC_iters,
                  const int basis_degree,
                  const arma::vec boundary_knots,
                  const arma::vec internal_knots,
                  int Warm_block = 500,
                  int Leapfrog_steps = 10,
                  const double sigma_A_mean = 6.32,
                  const double sigma_A_shape = 1,
                  const double sigma_B_mean = 6.32,
                  const double sigma_B_shape = 1,
                  const double alpha = 1,
                  const double beta = 0.005,
                  const double mu_prior_mean = 4,
                  const double mu_prior_var = 1,
                  const double eps_step = 0.001,
                  double step_size =  0.001,
                  double step_size_delta =  0.00005){
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
       time(j) = arma::accu(X_B(i,0).subvec(0,j));
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
       time(j) = arma::accu(X_AB(i,0).subvec(0,j));
     }
     bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                 boundary_knots);
     // Get Basis matrix
     arma::mat bspline_mat{bspline.basis(true)};
     basis_funct_AB(i,0) = bspline_mat;
   }
   
   
   
   Rcpp::List output = NeuralComp::HMC_sampler_TI(Labels, basis_funct_A, basis_funct_B, basis_funct_AB,
                                                  X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                  MCMC_iters, Leapfrog_steps, sigma_A_mean, sigma_A_shape,
                                                  sigma_B_mean, sigma_B_shape, alpha, beta, mu_prior_mean,
                                                  mu_prior_var, eps_step, step_size, Warm_block);
   return output;
   
 }

//[[Rcpp::export]]
Rcpp::List HMC_FR(arma::field<arma::vec> Labels,
                  const arma::field<arma::vec> X_A,
                  const arma::field<arma::vec> X_B,
                  const arma::field<arma::vec> X_AB,
                  const arma::vec n_A,
                  const arma::vec n_B,
                  const arma::vec n_AB,
                  int MCMC_iters,
                  const int basis_degree,
                  const arma::vec boundary_knots,
                  const arma::vec internal_knots,
                  int Warm_block = 500,
                  int Leapfrog_steps = 10,
                  const double I_A_shape = 40, 
                  const double I_A_rate = 1,
                  const double I_B_shape = 40,
                  const double I_B_rate = 1,
                  const double sigma_A_mean = 6.32,
                  const double sigma_A_shape = 1,
                  const double sigma_B_mean = 6.32,
                  const double sigma_B_shape = 1,
                  const double alpha = 0.1,
                  const double beta = 0.1,
                  const double mu_prior_mean = 4,
                  const double mu_prior_var = 0.1,
                  const double eps_step = 0.001,
                  double step_size_sigma =  0.001,
                  double step_size_FR =  0.1){
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
      time(j) = arma::accu(X_B(i,0).subvec(0,j));
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
      time(j) = arma::accu(X_AB(i,0).subvec(0,j));
    }
    bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix
    arma::mat bspline_mat{bspline.basis(true)};
    basis_funct_AB(i,0) = bspline_mat;
  }
  
  
  
  Rcpp::List output = NeuralComp::HMC_sampler_FR(Labels, basis_funct_A, basis_funct_B, basis_funct_AB,
                                                 X_A, X_B, X_AB, n_A, n_B, n_AB,
                                                 MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                                 I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                                 sigma_B_mean, sigma_B_shape, alpha, beta, mu_prior_mean,
                                                 mu_prior_var, eps_step, step_size_FR, step_size_sigma, Warm_block);
  return output;
  
}

//' MH sampler for labels
 //[[Rcpp::export]]
 arma::field<arma::vec> Sample_Labels(const arma::field<arma::vec> X_AB,
                                      const arma::vec n_AB,
                                      int MCMC_iters,
                                      arma::vec theta){
   
   arma::field<arma::vec> init_position(n_AB.n_elem, 1);
   for(int i = 0; i < n_AB.n_elem; i++){
     init_position(i,0) = arma::zeros(n_AB(i));
   }
   
   arma::field<arma::vec> Labels = NeuralComp::labels_sampler(X_AB, init_position,
                                                              n_AB, theta, MCMC_iters);
   return Labels;
   
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


//[[Rcpp::export]]
double posterior_Z1(arma::vec& Labels,
                    const arma::vec& X_AB,
                    arma::vec& theta,
                    int spike_num,
                    int n_AB){
  return NeuralComp::posterior_Z(Labels, X_AB, theta, spike_num, n_AB);
}


//[[Rcpp::export]]
Rcpp::List Sampler(const arma::field<arma::vec> X_A,
                   const arma::field<arma::vec> X_B,
                   const arma::field<arma::vec> X_AB,
                   const arma::vec n_A,
                   const arma::vec n_B,
                   const arma::vec n_AB,
                   int MCMC_iters,
                   int Warm_block = 500,
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
                   double step_size =  0.001,
                   double step_size_delta =  0.00005,
                   const double& step_size_labels = 0.0001,
                   const int& num_evals = 10000,
                   const double prior_p_labels = 0.5,
                   Rcpp::Nullable<Rcpp::NumericMatrix> Mass_mat = R_NilValue){
  arma::vec init_position1;
  if(init_position.isNull()){
    init_position1 = {50, 50, sqrt(50), sqrt(50), 0.15};
  }else{
    Rcpp::NumericVector X_(init_position);
    init_position1 = Rcpp::as<arma::vec>(X_);
  }
  arma::vec eps_step1;
  if(eps_step.isNull()){
    eps_step1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericVector X_(eps_step);
    eps_step1 = Rcpp::as<arma::vec>(X_);
  }
  arma::mat Mass_mat1;
  if(Mass_mat.isNull()){
    arma::vec diag_elem = {1, 1, 1, 1};
    Mass_mat1 = arma::diagmat(diag_elem);
  }else{
    Rcpp::NumericMatrix X_(Mass_mat);
    Mass_mat1 = Rcpp::as<arma::mat>(X_);
  }
  
  
  Rcpp::List param = NeuralComp::Total_sampler(X_A, X_B, X_AB, n_A, n_B, n_AB, init_position1,
                                               MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                               I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                               eps_step1, step_size, step_size_delta, step_size_labels,
                                               num_evals, prior_p_labels, Mass_mat1, 
                                               Warm_block);
  return param;
  
}

//[[Rcpp::export]]
Rcpp::List Mixed_Sampler(const arma::field<arma::vec> X_A,
                         const arma::field<arma::vec> X_B,
                         const arma::field<arma::vec> X_AB,
                         const arma::vec n_A,
                         const arma::vec n_B,
                         const arma::vec n_AB,
                         int MCMC_iters,
                         int Warm_block1 = 200,
                         int Warm_block2 = 300,
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
                         const double delta_shape = 0.01,
                         const double delta_rate = 0.1,
                         Rcpp::Nullable<Rcpp::NumericVector> eps_step = R_NilValue,
                         double step_size =  0.001,
                         double step_size_delta =  0.00002,
                         const double& step_size_labels = 0.0001,
                         const int& num_evals = 10000,
                         double delta_proposal_mean = -2,
                         double delta_proposal_sd = 0.3,
                         double alpha = 0.2,
                         int delta_adaption_block = 100,
                         int M_proposal = 10,
                         int n_Ensambler_sampler = 5,
                         Rcpp::Nullable<Rcpp::NumericMatrix> Mass_mat = R_NilValue){
  arma::vec init_position1;
  if(init_position.isNull()){
    init_position1 = {50, 50, sqrt(50), sqrt(50), 0.01};
  }else{
    Rcpp::NumericVector X_(init_position);
    init_position1 = Rcpp::as<arma::vec>(X_);
  }
  arma::vec eps_step1;
  if(eps_step.isNull()){
    eps_step1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericVector X_(eps_step);
    eps_step1 = Rcpp::as<arma::vec>(X_);
  }
  arma::mat Mass_mat1;
  if(Mass_mat.isNull()){
    arma::vec diag_elem = {1, 1, 1, 1};
    Mass_mat1 = arma::diagmat(diag_elem);
  }else{
    Rcpp::NumericMatrix X_(Mass_mat);
    Mass_mat1 = Rcpp::as<arma::mat>(X_);
  }
  
  
  Rcpp::List param = NeuralComp::Mixed_sampler(X_A, X_B, X_AB, n_A, n_B, n_AB, init_position1,
                                               MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                               I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                               sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                               eps_step1, step_size, step_size_delta, step_size_labels,
                                               num_evals, delta_proposal_mean, delta_proposal_sd, 
                                               alpha, delta_adaption_block,
                                               M_proposal, n_Ensambler_sampler, Mass_mat1, 
                                               Warm_block1, Warm_block2);
  return param;
  
}

//[[Rcpp::export]]
Rcpp::List Mixed_Sampler_int(const arma::field<arma::vec> X_A,
                             const arma::field<arma::vec> X_B,
                             const arma::field<arma::vec> X_AB,
                             const arma::vec n_A,
                             const arma::vec n_B,
                             const arma::vec n_AB,
                             int MCMC_iters,
                             int Warm_block1 = 200,
                             int Warm_block2 = 300,
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
                             const double delta_shape = 0.01,
                             const double delta_rate = 0.1,
                             Rcpp::Nullable<Rcpp::NumericVector> eps_step = R_NilValue,
                             double step_size =  0.001,
                             double step_size_delta =  0.00002,
                             const double& step_size_labels = 0.0001,
                             const int& num_evals = 10000,
                             double delta_proposal_mean = -2,
                             double delta_proposal_sd = 0.3,
                             double alpha = 0.2,
                             int delta_adaption_block = 100,
                             int M_proposal = 10,
                             int n_Ensambler_sampler = 5,
                             Rcpp::Nullable<Rcpp::NumericMatrix> Mass_mat = R_NilValue){
  arma::vec init_position1;
  if(init_position.isNull()){
    init_position1 = {50, 50, sqrt(50), sqrt(50), 0.01};
  }else{
    Rcpp::NumericVector X_(init_position);
    init_position1 = Rcpp::as<arma::vec>(X_);
  }
  arma::vec eps_step1;
  if(eps_step.isNull()){
    eps_step1 = {0.001, 0.001, 0.001, 0.001, 0.00005};
  }else{
    Rcpp::NumericVector X_(eps_step);
    eps_step1 = Rcpp::as<arma::vec>(X_);
  }
  arma::mat Mass_mat1;
  if(Mass_mat.isNull()){
    arma::vec diag_elem = {1, 1, 1, 1};
    Mass_mat1 = arma::diagmat(diag_elem);
  }else{
    Rcpp::NumericMatrix X_(Mass_mat);
    Mass_mat1 = Rcpp::as<arma::mat>(X_);
  }
  
  
  Rcpp::List param = NeuralComp::Mixed_sampler_int(X_A, X_B, X_AB, n_A, n_B, n_AB, init_position1,
                                                   MCMC_iters, Leapfrog_steps, I_A_shape, I_A_rate,
                                                   I_B_shape, I_B_rate, sigma_A_mean, sigma_A_shape,
                                                   sigma_B_mean, sigma_B_shape, delta_shape, delta_rate,
                                                   eps_step1, step_size, step_size_delta, step_size_labels,
                                                   num_evals, delta_proposal_mean, delta_proposal_sd, 
                                                   alpha, delta_adaption_block,
                                                   M_proposal, n_Ensambler_sampler, Mass_mat1, 
                                                   Warm_block1, Warm_block2);
  return param;
  
}

//[[Rcpp::export]]
arma::mat approx_trans_p(double step_size,
                         int num_evals,
                         arma::vec& theta){
  return NeuralComp::approx_trans_prob(step_size, num_evals, theta);
}

//[[Rcpp::export]]
arma::mat forward_pass1(arma::vec& theta,
                        const arma::vec& X_AB,
                        double step_size,
                        int num_evals){
  arma::vec theta_exp = arma::exp(theta);
  arma::vec theta_0 = theta_exp;
  theta_0(4) = 0;
  arma::mat trans_prob_0 = NeuralComp::approx_trans_prob(step_size, num_evals, theta_0);
  arma::mat trans_prob = NeuralComp::approx_trans_prob(step_size, num_evals, theta_exp);
  return NeuralComp::forward_pass(theta_exp, X_AB);
}

//[[Rcpp::export]]
arma::vec backward_sim1(arma::mat& Prob_mat,
                        arma::vec& theta,
                        const arma::vec& X_AB,
                        double step_size,
                        int num_evals){
  double prob_propose = 0;
  arma::vec theta_exp = arma::exp(theta);
  arma::vec ph = NeuralComp::backward_sim(Prob_mat, theta_exp, X_AB, prob_propose);
  Rcpp::Rcout << prob_propose;
  return ph;
}

//[[Rcpp::export]]
arma::field<arma::vec> FFBS_labels(const arma::field<arma::vec>& X_AB,
                                   const arma::vec& n_AB,
                                   arma::vec& theta,
                                   double step_size,
                                   int num_evals,
                                   int MCMC_iters){
  arma::field<arma::vec> init_position(n_AB.n_elem, 1);
  for(int i = 0; i < n_AB.n_elem; i++){
    init_position(i,0) = arma::zeros(n_AB(i));
  }
  
  arma::field<arma::vec> Labels = NeuralComp::FFBS(X_AB, init_position, n_AB, theta, step_size, 
                                                   num_evals, MCMC_iters);
  return Labels;
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
//[[Rcpp::export]]
arma::vec calc_gradient1(arma::field<arma::vec>& Labels,
                         arma::vec theta,
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
                         const arma::vec& eps_step){
  arma::vec grad(theta.n_elem, arma::fill::zeros);
  arma::vec theta_p_eps = theta;
  arma::vec theta_m_eps = theta;
  for(int i = 0; i < theta.n_elem; i++){
    theta_p_eps = theta;
    // f(x + e) in the i^th dimension
    theta_p_eps(i) = theta_p_eps(i) + eps_step(i);
    theta_m_eps = theta;
    // f(x - e) in the i^th dimension
    theta_m_eps(i) = theta_m_eps(i) - eps_step(i);
    // approximate gradient ((f(x + e) f(x - e))/ 2e)
    grad(i) = (NeuralComp::log_posterior(Labels, theta_p_eps, X_A, X_B, X_AB,
               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
               I_B_rate, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape) - NeuralComp::log_posterior(Labels, theta_m_eps, X_A, X_B, X_AB,
               n_A, n_B, n_AB, I_A_shape, I_A_rate, I_B_shape,
               I_B_rate, sigma_A_mean, sigma_A_shape,
               sigma_B_mean, sigma_B_shape)) / (2 * eps_step(i));

  }
  return grad;
}
// 
//[[Rcpp::export]]
double log_likelihood1(arma::field<arma::vec>& Labels,
                       arma::vec& theta,
                       const arma::field<arma::vec>& X_A,
                       const arma::field<arma::vec>& X_B,
                       const arma::field<arma::vec>& X_AB,
                       const arma::vec& n_A,
                       const arma::vec& n_B,
                       const arma::vec& n_AB){
  double l_likelihood = 0;

  // Calculate log-likelihood for A trials
  for(int i = 0; i < n_A.n_elem; i++){
    for(int j = 0; j < n_A(i); j++){
      l_likelihood = l_likelihood + NeuralComp::dinv_gauss(X_A(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
    }
  }

  // Calculate log-likelihood for B trials
  for(int i = 0; i < n_B.n_elem; i++){
    for(int j = 0; j < n_B(i); j++){
      l_likelihood = l_likelihood + NeuralComp::dinv_gauss(X_B(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
    }
  }

  // calculate log-likelihood for AB trials
  for(int i = 0; i < n_AB.n_elem; i++){
    for(int j = 0; j < n_AB(i); j++){
      if(Labels(i,0)(j) == 0){
        // label is A
        if(j != 0){
          if(Labels(i,0)(j-1) == 0){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2)) +
              NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
              NeuralComp::dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2));
          }
        }else{
          l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2)) +
            NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2));
        }
      }else{
        // label is B
        if(j != 0){
          if(Labels(i,0)(j-1) == 1){
            // Condition if spike has not switched (still in A)
            l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(0)), pow((1 / theta(2)), 2)) +
              NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
          }else{
            // Condition if spike has switched from B to A
            l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
              NeuralComp::dinv_gauss(X_AB(i,0)(j) - theta(4), (1 / theta(1)), pow((1 / theta(3)), 2));
          }
        }else{
          l_likelihood = l_likelihood + NeuralComp::pinv_gauss(X_AB(i,0)(j), (1 / theta(0)), pow((1 / theta(2)), 2)) +
            NeuralComp::dinv_gauss(X_AB(i,0)(j), (1 / theta(1)), pow((1 / theta(3)), 2));
        }
      }
      if(l_likelihood < -999999999999999){
        Rcpp::Rcout << i << j << "\n";
        i = 100;
        j = 100;
      }
    }
  }
  return l_likelihood;
}

//[[Rcpp::export]]
double rinv_gauss1(double mean,
                   double shape){
  return NeuralComp::rinv_gauss(mean, shape);
}

//[[Rcpp::export]]
arma::vec r_multinomial(arma::vec prob){
  return NeuralComp::rmutlinomial(prob);
}


//[[Rcpp::export]]
Rcpp::List FFBS_ensemble(const arma::field<arma::vec>& X_AB,
                         const arma::vec& n_AB,
                         arma::vec theta,
                         int MCMC_iters,
                         const double step_size = 0.0001,
                         const int num_evals = 10000,
                         double delta_proposal_mean = -2,
                         double delta_proposal_sd = 0.5,
                         int M_proposal = 10,
                         const double delta_shape= 0.5,
                         const double delta_rate = 0.1,
                         const double alpha = 0.2){
  arma::field<arma::vec> Labels(n_AB.n_elem, MCMC_iters);
  // Use initial starting position
  for(int j = 0; j < MCMC_iters; j++){
    for(int i = 0; i < n_AB.n_elem; i++){
      Labels(i, j) = arma::zeros(n_AB(i));
    }
  }
  arma::mat thetas(MCMC_iters, theta.n_elem, arma::fill::zeros);
  thetas.row(0) = theta.t();
  thetas.row(1) = theta.t();
  arma::vec theta_i(theta.n_elem, arma::fill::zeros);
  for(int i = 1; i < MCMC_iters; i++){
    theta_i = thetas.row(i).t();
    NeuralComp::FFBS_ensemble_step1(Labels, i, X_AB, n_AB, theta_i, step_size,
                                   num_evals, delta_proposal_mean, delta_proposal_sd, alpha, 
                                   M_proposal, delta_shape, delta_rate);
    if((i % 25) == 0){
      Rcpp::Rcout << "Iteration " << i;
    }
    
    thetas.row(i) = theta_i.t();
    if((i + 1) < MCMC_iters){
      thetas.row(i + 1) = thetas.row(i);
      for(int j = 0; j < n_AB.n_elem; j++){
        Labels(j, i + 1) = Labels(j, i);
      }
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("theta", arma::exp(thetas)),
                                         Rcpp::Named("labels", Labels));
  return params;
}

//[[Rcpp::export]]
arma::field<arma::vec> prior_Labels1(const arma::vec& n_AB,
                                     arma::mat trans_prob_0,
                                     arma::mat trans_prob){
  return NeuralComp::prior_Labels(n_AB, trans_prob_0, trans_prob);
}

//[[Rcpp::export]]
double calc_log_sum1(arma::vec x){
  return NeuralComp::calc_log_sum(x);
}

//[[Rcpp::export]]
double posterior_Labels1(arma::vec& Labels,
                         const arma::vec& X_AB,
                         arma::vec &theta){
  return NeuralComp::posterior_Labels(Labels, X_AB, theta);
}

//[[Rcpp::export]]
double prob_transition1(double label,
                        double label_next,
                        const arma::vec& X_AB,
                        arma::vec& theta,
                        int spike_num){
  return NeuralComp::prob_transition(label, label_next, X_AB, theta, spike_num);
}

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