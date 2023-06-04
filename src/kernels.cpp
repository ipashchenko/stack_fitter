#include "kernels.h"


Eigen::VectorXd gp_values_eigen(Eigen::VectorXd v, Eigen::VectorXd x, double amp, double scale)
{
	MatrixXd sqdist = - 2*x*x.transpose();
	sqdist.rowwise() += x.array().square().transpose().matrix();
	sqdist.colwise() += x.array().square().matrix();
	sqdist *= (-0.5/(scale*scale));
	MatrixXd C = amp * sqdist.array().exp();
	Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
	MatrixXd L = cholesky.matrixL();
	return L*v;
}

MatrixXd squared_exponential_kernel(VectorXd x, double amp, double scale)
{
	MatrixXd sqdist = - 2*x*x.transpose();
	sqdist.rowwise() += x.array().square().transpose().matrix();
	sqdist.colwise() += x.array().square().matrix();
	sqdist *= (-0.5/(scale*scale));
	MatrixXd C = amp * amp * sqdist.array().exp();
	return C;
}

MatrixXd non_stationary_squared_exponential_kernel(VectorXd x, double amp, double scale)
{
	return MatrixXd();
}

MatrixXd linear_kernel(VectorXd x, double amp_b, double amp_v, double c)
{
	VectorXd x_c = x.array() - c;
	MatrixXd sqdist = - 2*x_c*x_c.transpose();
	sqdist.rowwise() += x_c.array().square().transpose().matrix();
	sqdist.colwise() += x_c.array().square().matrix();
	sqdist *= amp_v*amp_v;
	MatrixXd C = amp_b*amp_b + sqdist.array();
	return C;
}

MatrixXd polynomial_kernel(VectorXd x, double power, double amp)
{
	MatrixXd sqdist = - 2*x*x.transpose();
	sqdist.rowwise() += x.array().square().transpose().matrix();
	sqdist.colwise() += x.array().square().matrix();
	MatrixXd C = pow(amp*amp + sqdist.array(), power);
	return C;
}
