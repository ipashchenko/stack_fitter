#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <exception>
#include <Eigen/Dense>

using Eigen::VectorXd;
//#include "armadillo"


class FailedDeterminantCalculationException : public std::exception {
		const char * what () const noexcept override {
			return "Failed to calculate the determinant!";
		}
};

std::vector<Eigen::Index> find_less(VectorXd X, double x);

std::vector<Eigen::Index> find_ge(VectorXd X, double x);

//double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df);

#endif //UTILS_H
