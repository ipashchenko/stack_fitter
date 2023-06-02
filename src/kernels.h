#ifndef KERNELS_H
#define KERNELS_H

#include "armadillo"


arma::vec gp_values_arma(arma::vec v, arma::vec times, double amp, double scale);

arma::mat squared_exponential_kernel(arma::vec x, double amp, double scale, double jitter = 1e-5);

arma::mat rational_quadratic_kernel(arma::vec x, double amp, double scale, double alpha, double jitter = 1e-5);

arma::mat periodic_kernel(arma::vec x, double amp, double scale, double period);

#endif //KERNELS_H
