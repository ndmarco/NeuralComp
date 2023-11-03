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
arma::mat approx_trans_p(double step_size,
                         int num_evals,
                         arma::vec& theta){
  return NeuralComp::approx_trans_prob(step_size, num_evals, theta);
}

// //[[Rcpp::export]]
// arma::mat forward_pass1(arma::vec& Labels,
//                         arma::vec& theta,
//                         const arma::vec& X_AB,
//                         double step_size,
//                         int num_evals){
//   return NeuralComp::forward_pass(theta, X_AB, step_size, num_evals);
// }

//[[Rcpp::export]]
arma::vec backward_sim1(arma::mat& Prob_mat,
                        arma::vec& theta,
                        const arma::vec& X_AB,
                        double step_size,
                        int num_evals){
  double prob_propose = 0;
  arma::vec ph = NeuralComp::backward_sim(Prob_mat, theta, X_AB, prob_propose);
  Rcpp::Rcout << prob_propose;
  return ph;
}

//[[Rcpp::export]]
arma::field<arma::vec> FFBS_labels(const arma::field<arma::vec>& X_AB,
                                   const arma::vec& n_AB,
                                   arma::vec& theta,
                                   double step_size,
                                   int num_evals,
                                   double prior_p_labels,
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
                         double delta_proposal_mean = 0.1,
                         double delta_proposal_shape = 0.05,
                         int M_proposal = 10,
                         const double delta_shape= 0.5,
                         const double delta_rate = 0.1){
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
    NeuralComp::FFBS_ensemble_step(Labels, i, X_AB, n_AB, theta_i, step_size,
                                   num_evals, delta_proposal_mean, delta_proposal_shape,
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