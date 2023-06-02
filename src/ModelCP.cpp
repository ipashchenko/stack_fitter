#include "ModelCP.h"
#include "Data.h"


using namespace DNest4;

ModelCP::ModelCP()
: a_before(0.0), a_after(0.0), b_before(0.0), r0(0.0), r1(0.0), changepoint(0.0),frac_log_error_scale(0.0),
abs_log_error_scale(0.0), log_dr(0.0), log_rmin(0.0), log_rmax(0.0)
{
	arma::vec r = Data::get_instance().get_r();
	mu = arma::zeros(r.size());
}

void ModelCP::from_prior(DNest4::RNG &rng)
{
	arma::vec r = Data::get_instance().get_r();
	double r_min = arma::min(r);
	double r_max = arma::max(r);
	log_rmin = log(r_min);
	log_rmax = log(r_max);
	log_dr = log(r_max) - log(r_min);
	
	b_before = -2.0 + 1.0*rng.randn();
	a_before = 1.0 + 0.25*rng.randn();
	a_after = 1.0 + 0.25*rng.randn();
	r0 = -2.0 + 1.0*rng.randn();
	// logU(xmin, xmax) => U(log(xmin), log(xmax))
//    changepoint = log(xmin) + (log(xmax) - log(xmin))*rng.rand();
	changepoint = log_rmin + log_dr*rng.rand();
	std::cout << "r_cp = " << exp(changepoint) << std::endl;
	r1 = 0.0 + 0.5*rng.randn();
}

double ModelCP::perturb(DNest4::RNG &rng)
{
	double logH = 0.0;
	double r = rng.rand();
	
	// Perturb b_before
	if(r <= 0.125) {
		logH -= -0.5*pow((b_before + 2.0)/1.0, 2.0);
		b_before += 1.0*rng.randh();
		logH += -0.5*pow((b_before + 2.0)/1.0, 2.0);
		
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
	
	// Perturb a_before
	else if(r > 0.125 && r <= 0.25) {
		logH -= -0.5*pow((a_before - 1.0)/0.25, 2.0);
		a_before += 0.25*rng.randh();
		logH += -0.5*pow((a_before - 1.0)/0.25, 2.0);
		
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
	
	// Perturb a_after
	else if(r > 0.25 && r <= 0.375) {
		logH -= -0.5*pow((a_after - 1.0)/0.25, 2.0);
		a_after += 0.25*rng.randh();
		logH += -0.5*pow((a_after - 1.0)/0.25, 2.0);
		
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
	
	// Perturb r1 making it smaller than changepoint (changepoint can't be closer to the core than r1)
	else if(r > 0.375 && r <= 0.5) {
		logH -= -0.5*pow((r1 + 0.0)/0.5, 2.0);
		double delta = 0.5*rng.randh();
		// FIXME: exp
//        while(r1 + delta >= exp(changepoint)) {
//            delta = 0.5*rng.randh();
//        }
		r1 += delta;
		logH += -0.5*pow((r1 + 0.0)/0.5, 2.0);
		
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
	
	// Perturb changepoint making it larger than r1 (changepoint can't be closer to the core than r1)
	else if(r > 0.5 && r <= 0.625) {


//        double delta;
//        while(true){
//            double cp = changepoint;
//            delta = logdx_data*rng.randh();
//            cp += delta;
//            wrap(cp, logxmin_data, logxmax_data);
//            // FIXME: exp
//            if(exp(cp) >= r1){
//                break;
//            }
//        }
//        changepoint += delta;
		
		
		changepoint += log_dr*rng.randh();
		wrap(changepoint, log_rmin, log_rmax);
		calculate_prediction();
	}
	
	// Perturb scale error
	else if(r > 0.625 && r <= 0.75) {
		
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
	else if(r > 0.75 && r <= 0.875) {
		
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

void ModelCP::calculate_prediction()
{
	arma::vec r = Data::get_instance().get_r();
	double r_cp = exp(changepoint);
	arma::uvec before = arma::find(r < r_cp);
	arma::uvec after = arma::find(r >= r_cp);
	
	// Continuity
	double b_after = b_before + a_before*log(r_cp + exp(r0)) - a_after*log(r_cp + r1);
	
	mu(before) = exp(b_before) * arma::pow(r(before) + exp(r0), a_before);
	mu(after) = exp(b_after) * arma::pow(r(after) + r1, a_after);
}

double ModelCP::log_likelihood() const
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

void ModelCP::print(std::ostream &out) const
{
	out << a_before << "\t";
	out << a_after << "\t";
	out << b_before << "\t";
	double b_after = b_before + a_before*log(exp(changepoint) + exp(r0)) - a_after*log(exp(changepoint) + r1);
	out << b_after << "\t";
	out << r0 << "\t";
	out << r1 << "\t";
	out << changepoint << "\t";
	out << abs_log_error_scale << "\t";
	out << frac_log_error_scale;
}

std::string ModelCP::description() const
{
	std::string descr;
	descr += "a_before ";
	descr += "a_after ";
	descr += "b_before ";
	descr += "b_after ";
	descr += "r0 ";
	descr += "r1 ";
	descr += "log_changepoint ";
	descr += "abs_log_error_scale ";
	descr += "frac_log_error_scale";
	
	return descr;
}
