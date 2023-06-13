#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <exception>
#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>

using Eigen::VectorXd;
using Eigen::MatrixXd;
//#include "armadillo"


class FailedDeterminantCalculationException : public std::exception {
		const char * what () const noexcept override {
			return "Failed to calculate the determinant!";
		}
};

std::vector<Eigen::Index> find_less(VectorXd X, double x);

std::vector<Eigen::Index> find_ge(VectorXd X, double x);

std::vector<Eigen::Index> find_between(VectorXd X, double x_low, double x_high);

double profile(double z, double b, double shift, double a);

double profile_der(double z, double b, double shift, double a);

VectorXd profile(VectorXd z, double b, double shift, double a);

VectorXd profile_cp(VectorXd z, double z_0, double z_1, double z_br, double k_b, double k_a, double b_b, double dz);

//double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df);

#endif //UTILS_H
