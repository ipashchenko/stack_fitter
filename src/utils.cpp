#include <vector>
#include "utils.h"


//double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df)
//{
//	double np = 0.5*(df + 1);
//	double fac = log(tgamma(np)) - log(tgamma(0.5*df));
//	return arma::sum(-np*arma::log(1. + (x-mu)%(x-mu)/(sigma*sigma)/df) + fac - 0.5*log(M_PI*df) - log(sigma));
//}

std::vector<Eigen::Index> find_less(VectorXd X, double x)
{
	std::vector<double> result;
	std::vector<Eigen::Index> idxs;
	for(Eigen::Index i=0; i<X.size(); ++i)
		if(X(i) < x)
			idxs.push_back(i);
//			result.push_back(X(i));
//	VectorXd res = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(result.data(), result.size());
	return idxs;
}

std::vector<Eigen::Index> find_ge(VectorXd X, double x)
{
	std::vector<double> result;
	std::vector<Eigen::Index> idxs;
	for(Eigen::Index i=0; i<X.size(); ++i)
		if(X(i) >= x)
			idxs.push_back(i);
//			result.push_back(X(i));
//	VectorXd res = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(result.data(), result.size());
	return idxs;
}


std::vector<Eigen::Index> find_between(VectorXd X, double x_low, double x_high)
{
	std::vector<double> result;
	std::vector<Eigen::Index> idxs;
	for(Eigen::Index i=0; i<X.size(); ++i)
		if(X(i) > x_low && X(i) <= x_high)
			idxs.push_back(i);
//			result.push_back(X(i));
//	VectorXd res = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(result.data(), result.size());
	return idxs;
}


double profile(double z, double b, double shift, double a)
{
	return b*pow(z + shift, a);
}

double profile_der(double z, double b, double shift, double a)
{
	return a*b*pow(z + shift, a-1.);
}

VectorXd profile(VectorXd z, double b, double shift, double a)
{
	return b*pow(z.array() + shift, a);
}

VectorXd profile_cp(VectorXd z, double z_0, double z_1, double z_br, double k_b, double k_a, double b_b, double dz)
{
	VectorXd f1 = profile(z, b_b, z_0, k_b);
	double b_a = b_b*pow(z_br + z_0, k_b)/pow(z_br + z_1, k_a);
	VectorXd f2 = profile(z, b_a, z_1, k_a);
	
	double z_before = z_br - 0.5*dz;
	double z_after = z_br + 0.5*dz;
	double C1 = profile(z_before, b_b, z_0, k_b);
	double C2 = profile(z_after, b_a, z_1, k_a);
	double C3 = profile_der(z_before, b_b, z_0, k_b);
	double C4 = profile_der(z_after, b_a, z_1, k_a);
	
	Eigen::Vector4d b(4);
	b << C1, C2, C3, C4;
	Eigen::Matrix4d A;
	A  << pow(z_before, 3), pow(z_before, 2), z_before, 1,
	      pow(z_after,3), pow(z_after, 2), z_after, 1,
	      3*pow(z_before, 2), 2*z_before, 1, 0,
	      3*pow(z_after, 2), 2*z_after, 1, 0;
	// FIXME: Check that it does work
	Eigen::Vector4d coeffs = A.lu().solve(b);
	
	VectorXd poly = coeffs[0]*pow(z.array(), 3.) + coeffs[1]*pow(z.array(), 2.) + coeffs[2]*z.array() + coeffs[3];
	
	std::vector<Eigen::Index> before = find_less(z, z_before);
//	between = np.logical_and(z > z_before, z <= z_after)
	std::vector<Eigen::Index> between = find_between(z, z_before, z_after);
	
	VectorXd result = f2;
	result(before) = f1(before);
	result(between) = poly(between);
	
	return result;
}
