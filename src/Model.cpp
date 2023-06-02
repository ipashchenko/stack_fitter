#include "Model.h"
#include "Data.h"
#include "utils.h"

using namespace DNest4;


Model::Model()
: a(0.0), b(0.0), r0(0.0), frac_log_error_scale(0.0), abs_log_error_scale(0.0)
{
	arma::vec r = Data::get_instance().get_r();
	mu = arma::zeros(r.size());
}

void Model::from_prior(DNest4::RNG &rng)
{

	b = -2.0 + 1.0*rng.randn();
	a = 1.0 + 0.5*rng.randn();
	r0 = -2.0 + 1.0*rng.randn();
	frac_log_error_scale = -4.0 + 1.0 * rng.randn();
	abs_log_error_scale = -10.0 + 1.0 * rng.randn();
}

double Model::perturb(DNest4::RNG &rng)
{
	double logH = 0.0;
	double r = rng.rand();
	
	// Perturb b
	if(r <= 0.2) {
		logH -= -0.5*pow((b + 2.0)/1.0, 2.0);
		b += 1.0*rng.randh();
		logH += -0.5*pow((b + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH)) {
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb a
	else if(r > 0.2 && r <=0.4) {
		logH -= -0.5*pow((a - 1.0)/0.5, 2.0);
		a += 0.5*rng.randh();
		logH += -0.5*pow((a - 1.0)/0.5, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH)) {
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb scale error
	else if(r > 0.4 && r <= 0.6) {
		
		logH -= -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
		frac_log_error_scale += 1.0 * rng.randh();
		logH += -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
		// Pre-reject
		if(rng.rand() >= exp(logH)) {
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Perturb additive error
	else if(r > 0.6 && r <= 0.8) {
		
		logH -= -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
		abs_log_error_scale += 1.0 * rng.randh();
		logH += -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
		// Pre-reject
		if(rng.rand() >= exp(logH)) {
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Perturb r0
	else {
		logH -= -0.5*pow((r0 + 2.0)/1.0, 2.0);
		r0 += 1.0*rng.randh();
		logH += -0.5*pow((r0 + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH)) {
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	return logH;
}

void Model::calculate_prediction()
{
	arma::vec r = Data::get_instance().get_r();
	mu = exp(b)*pow(r + exp(r0), a);
}

double Model::log_likelihood() const
{
	double loglik = 0.0;
	arma::vec r = Data::get_instance().get_r();
	arma::vec R = Data::get_instance().get_R();
	arma::vec sigma = arma::sqrt(mu%mu*exp(2*frac_log_error_scale) + exp(2*abs_log_error_scale));
	// likelihood of Radius
	// Student-T
//	loglik += studentT_lpdf(R, mu, sigma, 2.0);
//	// Normal
	loglik += arma::sum(-0.5*log(2*M_PI*sigma%sigma) - 0.5*(arma::pow((R - mu)/sigma, 2.0)));
	return loglik;
}

void Model::print(std::ostream &out) const
{
	out << a << "\t";
	out << b << "\t";
	out << r0 << "\t";
	out << abs_log_error_scale << "\t";
	out << frac_log_error_scale;
}

std::string Model::description() const
{
	std::string descr;
	descr += "a ";
	descr += "b ";
	descr += "r0 ";
	descr += "abs_log_error_scale ";
	descr += "frac_log_error_scale";
	
	return descr;
}
