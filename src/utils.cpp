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