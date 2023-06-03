#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Core>
#include <Eigen/Eigen>

using Eigen::VectorXd;
using Eigen::MatrixXd;


VectorXd gp_values_eigen(VectorXd v, VectorXd x, double amp, double scale);

MatrixXd squared_exponential_kernel(VectorXd x, double amp, double scale);

#endif //KERNELS_H
