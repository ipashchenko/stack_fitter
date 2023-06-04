#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Core>
#include <Eigen/Eigen>

using Eigen::VectorXd;
using Eigen::MatrixXd;


VectorXd gp_values_eigen(VectorXd v, VectorXd x, double amp, double scale);

MatrixXd squared_exponential_kernel(VectorXd x, double amp, double scale);

MatrixXd non_stationary_squared_exponential_kernel(VectorXd x, double amp, double scale);

// See the Kernel cookbook
MatrixXd linear_kernel(VectorXd x, double amp_b, double amp_v, double c);

MatrixXd polynomial_kernel(VectorXd x, double power, double amp);

#endif //KERNELS_H
