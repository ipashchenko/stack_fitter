#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <exception>
//#include "armadillo"


class FailedDeterminantCalculationException : public std::exception {
		const char * what () const noexcept override {
			return "Failed to calculate the determinant!";
		}
};

//double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df);

#endif //UTILS_H
