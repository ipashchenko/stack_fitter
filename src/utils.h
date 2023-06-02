#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include "armadillo"


double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df);

#endif //UTILS_H
