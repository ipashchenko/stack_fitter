#include "ModelGP.h"
#include "Data.h"
#include "utils.h"
#include "kernels.h"

using namespace DNest4;
using std::pow;


ModelGP::ModelGP()
: a(0.0), b(0.0), r0(0.0), frac_log_error_scale(0.0), abs_log_error_scale(0.0), gp_logamp(0.0), gp_scale(1.0), gp_logalpha(0.0)
{
	VectorXd r = Data::get_instance().get_r();
	mu = VectorXd::Zero(r.size());
}

void ModelGP::from_prior(DNest4::RNG &rng)
{
	VectorXd r = Data::get_instance().get_r();

	DNest4::Gaussian gaussian(-2., 0.2);
	DNest4::TruncatedCauchy halfcauchy(0.0, 0.1);
	DNest4::Uniform uniform(-5., 5.0);
	b = 0.0 + 1.0*rng.randn();
	a = 1.0 + 0.5*rng.randn();
	r0 = -2.0 + 1.0*rng.randn();
	frac_log_error_scale = -10.0 + 1.0 * rng.randn();
	abs_log_error_scale = -10.0 + 1.0 * rng.randn();
	gp_logamp = gaussian.generate(rng);
	gp_scale = halfcauchy.generate(rng);
	gp_logalpha = uniform.generate(rng);
}

double ModelGP::perturb(DNest4::RNG &rng)
{
	double logH = 0.0;
	double r = rng.rand();
	
	// Perturb b
	if(r <= 0.175)
	{
		logH -= -0.5*pow((b + 2.0)/1.0, 2.0);
		b += 1.0*rng.randh();
		logH += -0.5*pow((b + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb a
	else if(r > 0.175 && r <=0.35)
	{
		logH -= -0.5*pow((a - 1.0)/0.5, 2.0);
		a += 0.5*rng.randh();
		logH += -0.5*pow((a - 1.0)/0.5, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
//	// Perturb scale error
//	else if(r > 0.4 && r <= 0.6)
//	{
//
//		logH -= -0.5*pow((frac_log_error_scale + 10.0) / 1.0, 2.0);
//		frac_log_error_scale += 1.0 * rng.randh();
//		logH += -0.5*pow((frac_log_error_scale + 10.0) / 1.0, 2.0);
//		// Pre-reject
//		if(rng.rand() >= exp(logH))
//		{
//			return -1E300;
//		}
//		else {
//			logH = 0.0;
//		}
//		// No need to re-calculate model. Just calculate loglike.
//	}
//
//	// Perturb additive error
//	else if(r > 0.6 && r <= 0.8)
//	{
//
//		logH -= -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
//		abs_log_error_scale += 1.0 * rng.randh();
//		logH += -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
//		// Pre-reject
//		if(rng.rand() >= exp(logH))
//		{
//			return -1E300;
//		}
//		else {
//			logH = 0.0;
//		}
//		// No need to re-calculate model. Just calculate loglike.
//	}
	
	// Perturb r0
	else if(r > 0.35 && r <= 0.525)
	{
		logH -= -0.5*pow((r0 + 2.0)/1.0, 2.0);
		r0 += 1.0*rng.randh();
		logH += -0.5*pow((r0 + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	// Scale
	else if(r > 0.525 && r <= 0.7)
	{
		DNest4::TruncatedCauchy halfcauchy(0.0, 0.1);
		logH += halfcauchy.perturb(gp_scale, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Alpha
	else if(r > 0.7 && r <= 0.875)
	{
		DNest4::Uniform uniform(-5., 5.0);
		logH += uniform.perturb(gp_logalpha, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	// Amplitude
	else
	{
		DNest4::Gaussian gaussian(-2., 0.2);
		logH += gaussian.perturb(gp_logamp, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	
	return logH;
}

void ModelGP::calculate_prediction()
{
	VectorXd r = Data::get_instance().get_r();
	mu = exp(b)*pow(r.array() + exp(r0), a);
}

double ModelGP::log_likelihood() const
{
	double loglik = 0.0;
	VectorXd r = Data::get_instance().get_r();
	VectorXd R = Data::get_instance().get_R();
	auto n = r.size();
	VectorXd diff = R - mu;
//	VectorXd disp = (mu.cwiseProduct(mu)*exp(2*frac_log_error_scale)).array() + exp(2*abs_log_error_scale);
	
//	MatrixXd C = squared_exponential_kernel(r, exp(gp_logamp), gp_scale);
	MatrixXd C = rational_quadratic_kernel(r, exp(gp_logamp), gp_scale, exp(gp_logalpha));
//	MatrixXd C_ = linear_kernel(r, 1e-05, 0.01, -exp(r0));
//	MatrixXd CC = C.array()*C_.array();
//	CC += disp.asDiagonal();
//	C += disp.asDiagonal();
	
//	https://stackoverflow.com/a/39735211
	Eigen::LLT<Eigen::MatrixXd> llt = C.llt();
	double sqrt_det = llt.matrixL().determinant();
	// likelihood of Radius
	loglik = -0.5*n*log(2*M_PI) - log(sqrt_det) - 0.5*(llt.matrixL().solve(diff)).squaredNorm();

	return loglik;
}

void ModelGP::print(std::ostream &out) const
{
	out << a << "\t";
	out << b << "\t";
	out << r0 << "\t";
	out << abs_log_error_scale << "\t";
	out << frac_log_error_scale << "\t";
	out << gp_logamp << "\t";
	out << gp_scale << "\t";
	out << gp_logalpha;
}

std::string ModelGP::description() const
{
	std::string descr;
	descr += "a ";
	descr += "b ";
	descr += "r0 ";
	descr += "abs_log_error_scale ";
	descr += "frac_log_error_scale ";
	descr += "gp_logamp ";
	descr += "gp_scale ";
	descr += "gp_logalpha";
	
	return descr;
}
