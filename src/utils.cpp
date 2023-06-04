#include "utils.h"


double studentT_lpdf(double x, double df, double mu, double sigma) {
	double np = 0.5*(df + 1);
	double fac = log(tgamma(np)) - log(tgamma(0.5*df));
	return -np*log(1. + (x-mu)*(x-mu)/(sigma*sigma)/df) + fac - 0.5*log(M_PI*df) - log(sigma);
}

double studentT_lpdf(VectorXd x, VectorXd mu, VectorXd disp, double df)
{
	double np = 0.5*(df + 1);
	double fac = log(tgamma(np)) - log(tgamma(0.5*df));
	return (-np*log(1. + (x-mu).array()*(x-mu).array()/disp.array()/df) + fac - 0.5*log(M_PI*df) - 0.5*log(disp.array())).sum();
}

double normal_lpdf(VectorXd x, VectorXd mu, VectorXd disp)
{
	return (-0.5*log(2*M_PI*disp.array()) - 0.5*((x-mu).array()*(x-mu).array()/disp.array())).sum();
}

//double studentT_lpdf(arma::vec x, arma::vec mu_model, arma::vec sigma, double df)
//{
//	double np = 0.5*(df + 1);
//	double fac = log(tgamma(np)) - log(tgamma(0.5*df));
//	return arma::sum(-np*arma::log(1. + (x-mu_model)%(x-mu_model)/(sigma*sigma)/df) + fac - 0.5*log(M_PI*df) - log(sigma));
//}
