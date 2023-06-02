#include "kernels.h"


arma::vec gp_values_arma(arma::vec v, arma::vec times, double amp, double scale)
{
	arma::mat sqdist = -2*times*times.t();
	sqdist.each_row() += arma::square(times).t();
	sqdist.each_col() += arma::square(times);
	sqdist *= (-0.5/(scale*scale));
	arma::mat C = amp * arma::exp(sqdist);
	arma::mat L = arma::chol(C, "lower");
	return L*v;
}

arma::mat squared_exponential_kernel(arma::vec x, double amp, double scale, double jitter)
{
	arma::mat sqdist = -2*x*x.t();
	sqdist.each_row() += arma::square(x).t();
	sqdist.each_col() += arma::square(x);
	sqdist *= (-0.5/(scale*scale));
	arma::mat C = amp*amp*arma::exp(sqdist);
	C += jitter*arma::eye(C.n_rows, C.n_cols);
	return C;
}

arma::mat rational_quadratic_kernel(arma::vec x, double amp, double scale, double alpha, double jitter)
{
	arma::mat sqdist = -2*x*x.t();
	sqdist.each_row() += arma::square(x).t();
	sqdist.each_col() += arma::square(x);
	sqdist *= 0.5/(alpha*scale*scale);
	arma::mat C = amp*amp*arma::pow(1. + sqdist, -alpha);
	C += jitter*arma::eye(C.n_rows, C.n_cols);
	return C;
}

arma::mat periodic_kernel(arma::vec x, double amp, double scale, double period)
{
	arma::mat sqdist = -2*x*x.t();
	sqdist.each_row() += arma::square(x).t();
	sqdist.each_col() += arma::square(x);
	sqdist = arma::sin(arma::datum::pi*arma::sqrt(sqdist)/period);
	arma::mat C = amp*amp*arma::exp(-2.*sqdist*sqdist/(scale*scale));
	return C;
}

