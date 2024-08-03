#ifndef NeuralComp_Unimodality_H
#define NeuralComp_Unimodality_H

#include <RcppArmadillo.h>
#include <cmath>
#include <CppAD.h>
#include "Sample.h"
using namespace CppAD;
using namespace Eigen;

namespace NeuralComp {

inline arma::vec find_peaks(arma::vec x){
  arma::vec x_diff = arma::diff(x);
  arma::vec peaks = arma::zeros(1);
  int n_peaks = 1;
  for(int i = 0; i < (x_diff.n_elem - 1); i++){
    if((x_diff(i) > 0) && (x_diff(i+1) < 0)){
      peaks.resize(n_peaks);
      peaks(n_peaks - 1) = i + 1;
      n_peaks = n_peaks + 1;
    }
  }
  return peaks;
}

inline arma::vec gaussian_KDE(const arma::vec& eval_grid,
                              const arma::vec& obs,
                              const double h){
  arma::vec density = arma::zeros(eval_grid.n_elem);
  for(int j = 0; j < obs.n_elem; j++){
    density = density + arma::normpdf(eval_grid, obs(j), h);
  }
  density = density / (obs.n_elem * h);
  return density;
}

inline int get_peaks_from_bw(const arma::vec eval_grid,
                             const arma::vec obs,
                             const double h){
  arma::vec den = gaussian_KDE(eval_grid, obs, h);
  arma::vec peaks = find_peaks(den);
  return peaks.n_elem;
}

inline arma::vec generate_bootstrap_dat(const arma::vec& dat,
                                        double h_crit){
  double sigma_sq = arma::var(dat);
  double x_mean = arma::mean(dat);
  arma::vec prob = arma::ones(dat.n_elem) / dat.n_elem;
  arma::vec x = sample(dat, dat.n_elem, true, prob);
  //arma::vec y = (x + h_crit * arma::randn(dat.n_elem)) / std::sqrt(1 + (h_crit * h_crit)/sigma_sq);
  arma::vec y = x_mean + (x - x_mean + h_crit * arma::randn(dat.n_elem)) / std::sqrt(1 + 1);
  arma::vec y_int = arma::round(y);
  return y_int;
}

inline double bootstrap_test_unimodality(const arma::vec obs_dat, 
                                         const arma::vec eval_grid,
                                         const arma::vec h_grid,
                                         const int n_boot){
  double p_val = 1;
  if(arma::var(obs_dat) > 0){
    double h_crit = 0.01;
    double peaks_i;
    for(int i = h_grid.n_elem - 1; i >= 0; i--){
      peaks_i = get_peaks_from_bw(eval_grid, obs_dat, h_grid(i));
      if(peaks_i > 1){
        if((i + 2) > h_grid.n_elem){
          Rcpp::stop("'h_grid' needs to include larger values.");
        }
        h_crit = h_grid(i + 1);
        break;
      }
    }
    Rcpp::Rcout << h_crit;
    int n_greater_1 = 0;
    int n_peaks_i = 0;
    arma::vec boot_x = arma::zeros(obs_dat.n_elem);
    // Conduct parametric smoothed bootstrap
    for(int i = 0; i < n_boot; i++){
      boot_x = generate_bootstrap_dat(obs_dat, h_crit);
      n_peaks_i = get_peaks_from_bw(eval_grid, boot_x, h_crit);
      if(n_peaks_i > 1){
        n_greater_1 = n_greater_1 + 1;
      }
    }
    p_val = n_greater_1 / (double)n_boot;
  }
  return p_val;
}

inline arma::vec generate_bootstrap_dat_ISI(const arma::vec& dat,
                                            double h_crit){
  double sigma_sq = arma::var(dat);
  arma::vec prob = arma::ones(dat.n_elem) / dat.n_elem;
  arma::vec x = sample(dat, dat.n_elem, true, prob);
  arma::vec y = (x + h_crit * arma::randn(dat.n_elem)) / std::sqrt(1 + (h_crit * h_crit)/sigma_sq);
  return y;
}

inline double bootstrap_test_unimodality_ISI(const arma::vec obs_dat, 
                                             const arma::vec eval_grid,
                                             const arma::vec h_grid,
                                             const int n_boot){
  double p_val = 1;
  if(arma::var(obs_dat) > 0){
    double h_crit = 0.01;
    double peaks_i;
    for(int i = h_grid.n_elem - 1; i >= 0; i--){
      peaks_i = get_peaks_from_bw(eval_grid, obs_dat, h_grid(i));
      if(peaks_i > 1){
        if((i + 2) > h_grid.n_elem){
          Rcpp::stop("'h_grid' needs to include larger values.");
        }
        h_crit = h_grid(i + 1);
        break;
      }
    }
    int n_greater_1 = 0;
    int n_peaks_i = 0;
    arma::vec boot_x = arma::zeros(obs_dat.n_elem);
    // Conduct parametric smoothed bootstrap
    for(int i = 0; i < n_boot; i++){
      boot_x = generate_bootstrap_dat_ISI(obs_dat, h_crit);
      n_peaks_i = get_peaks_from_bw(eval_grid, boot_x, h_crit);
      if(n_peaks_i > 1){
        n_greater_1 = n_greater_1 + 1;
      }
    }
    p_val = n_greater_1 / (double)n_boot;
  }
  return p_val;
}

}

#endif