#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <exception>
#include <Eigen/Dense>

using Eigen::VectorXd;


class FailedDeterminantCalculationException : public std::exception {
		const char * what () const noexcept override {
			return "Failed to calculate the determinant!";
		}
};

double studentT_lpdf(double x, double df, double mu, double sigma);

double studentT_lpdf(VectorXd x, VectorXd mu, VectorXd disp, double df);

double normal_lpdf(VectorXd x, VectorXd mu, VectorXd disp);


#endif //UTILS_H
