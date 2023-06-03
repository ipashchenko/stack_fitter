#include "kernels.h"


Eigen::VectorXd gp_values_eigen(Eigen::VectorXd v, Eigen::VectorXd x, double amp, double scale)
{
	Eigen::MatrixXd sqdist = - 2*x*x.transpose();
	sqdist.rowwise() += x.array().square().transpose().matrix();
	sqdist.colwise() += x.array().square().matrix();
	sqdist *= (-0.5/(scale*scale));
	Eigen::MatrixXd C = amp * sqdist.array().exp();
	Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
	Eigen::MatrixXd L = cholesky.matrixL();
	return L*v;
}

MatrixXd squared_exponential_kernel(VectorXd x, double amp, double scale)
{
	Eigen::MatrixXd sqdist = - 2*x*x.transpose();
	sqdist.rowwise() += x.array().square().transpose().matrix();
	sqdist.colwise() += x.array().square().matrix();
	sqdist *= (-0.5/(scale*scale));
	Eigen::MatrixXd C = amp * amp * sqdist.array().exp();
	return C;
}
