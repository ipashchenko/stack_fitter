#include "utils.h"


double studentT_lpdf(arma::vec x, arma::vec mu, arma::vec sigma, double df)
{
	double np = 0.5*(df + 1);
	double fac = log(tgamma(np)) - log(tgamma(0.5*df));
	return arma::sum(-np*arma::log(1. + (x-mu)%(x-mu)/(sigma*sigma)/df) + fac - 0.5*log(M_PI*df) - log(sigma));
}
