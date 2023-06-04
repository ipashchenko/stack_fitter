#include "ModelGPLatent.h"
#include "Data.h"
#include "utils.h"
#include "kernels.h"

using namespace DNest4;
using std::pow;


ModelGPLatent::ModelGPLatent()
: a(0.0), b(0.0), r0(0.0), frac_log_error_scale(0.0), abs_log_error_scale(0.0), gp_logamp(0.0), gp_scale(1.0)
{
	VectorXd r = Data::get_instance().get_r();
	v = VectorXd::Zero(r.size());
	C = MatrixXd::Ones(r.size(), r.size());
	L = MatrixXd::Ones(r.size(), r.size());
	mu_gp = VectorXd::Zero(r.size());
	mu_model = VectorXd::Zero(r.size());
}

void ModelGPLatent::from_prior(DNest4::RNG &rng)
{
	VectorXd r = Data::get_instance().get_r();

	DNest4::Gaussian gaussian(-2., 1.0);
	b = -2.0 + 1.0*rng.randn();
	a = 1.0 + 0.5*rng.randn();
	r0 = -2.0 + 1.0*rng.randn();
	frac_log_error_scale = -4.0 + 1.0 * rng.randn();
	abs_log_error_scale = -10.0 + 1.0 * rng.randn();
	gp_logamp = gaussian.generate(rng);
}

double ModelGPLatent::perturb(DNest4::RNG &rng)
{
	double logH = 0.0;
	double r = rng.rand();
	
	// Perturb b
	if(r <= 0.2)
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
		calculate_model_prediction();
	}
	
	// Perturb a
	else if(r > 0.2 && r <=0.4)
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
		calculate_model_prediction();
	}
	
	// Perturb scale error
	else if(r > 0.4 && r <= 0.6)
	{
		
		logH -= -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
		frac_log_error_scale += 1.0 * rng.randh();
		logH += -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Perturb additive error
	else if(r > 0.6 && r <= 0.8)
	{
		
		logH -= -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
		abs_log_error_scale += 1.0 * rng.randh();
		logH += -0.5*pow((abs_log_error_scale + 10.0) / 1.0, 2.0);
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Perturb r0
	else if(r > 0.8 && r <= 0.9)
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
		calculate_model_prediction();
	}
	else
	{
		DNest4::Gaussian gaussian(-2., 1.0);
		logH += gaussian.perturb(gp_logamp, rng);
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else {
			logH = 0.0;
		}
		// No need to re-calculate model. Just calculate loglike.
	}
	
	return logH;
}


void ModelGPLatent::calculate_C()
{
	VectorXd r = Data::get_instance().get_r();
	C = squared_exponential_kernel(r, exp(gp_logamp), gp_scale);
}

void ModelGPLatent::calculate_L()
{
	// perform the Cholesky decomposition of covariance matrix
	Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
	// get the lower triangular matrix L
	L = cholesky.matrixL();
}

void ModelGPLatent::calculate_gp_prediction()
{
	mu_gp = L*v;
}

void ModelGPLatent::calculate_model_prediction()
{
	VectorXd r = Data::get_instance().get_r();
	mu_model = exp(b)*pow(r.array() + exp(r0), a);
}

double ModelGPLatent::log_likelihood() const
{
	double loglik;
	VectorXd r = Data::get_instance().get_r();
	VectorXd R = Data::get_instance().get_R();
	// Full prediction
	VectorXd mu = mu_model + mu_gp;
	// Errors
	VectorXd disp = (mu.cwiseProduct(mu)*exp(2*frac_log_error_scale)).array() + exp(2*abs_log_error_scale);
	
	loglik = studentT_lpdf(R, mu_model+mu_gp, disp, 2.0);
	
	return loglik;
}

void ModelGPLatent::print(std::ostream &out) const
{
	out << a << "\t";
	out << b << "\t";
	out << r0 << "\t";
	out << abs_log_error_scale << "\t";
	out << frac_log_error_scale << "\t";
	out << gp_logamp;
}

std::string ModelGPLatent::description() const
{
	std::string descr;
	descr += "a ";
	descr += "b ";
	descr += "r0 ";
	descr += "abs_log_error_scale ";
	descr += "frac_log_error_scale ";
	descr += "gp_logamp";
	
	return descr;
}
