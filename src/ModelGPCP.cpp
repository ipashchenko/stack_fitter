#include "ModelGPCP.h"
#include "Data.h"
#include "utils.h"
#include "kernels.h"

using namespace DNest4;
using std::pow;


ModelGPCP::ModelGPCP()
: a_before(0.0), a_after(0.0), b_before(0.0), r0(0.0), r1(0.0), changepoint(0.0),frac_log_error_scale(0.0),
  abs_log_error_scale(0.0), log_dr(0.0), log_rmin(0.0), log_rmax(0.0), gp_logamp(0.0), gp_scale(1.0), gp_logalpha(0.0)
{
	VectorXd r = Data::get_instance().get_r();
	mu = VectorXd::Zero(r.size());
}

void ModelGPCP::from_prior(DNest4::RNG &rng)
{
	VectorXd r = Data::get_instance().get_r();
	VectorXd R = Data::get_instance().get_R();
	std::cout << "\n r = " << r << "\n";
	std::cout << "\n R = " << R << "\n";
	double r_min = r.minCoeff();
	double r_max = r.maxCoeff();
	log_rmin = log(r_min);
	log_rmax = log(r_max);
	log_dr = log(r_max) - log(r_min);
	
	std::cout << "r_min = " << r_min << ", r_max = " << r_max << "\n";
	std::cout << "log_dr = " << log_dr << "\n";

	// This works with x10 data
//	DNest4::Gaussian gaussian(-2., 1.0);
	DNest4::Gaussian gaussian(-2., 0.2);
	DNest4::Cauchy halfcauchy(0.0, 0.1);
	DNest4::Uniform uniform(-5., 5.0);
	b_before = -2.0 + 1.0*rng.randn();
	a_before = 1.0 + 0.5*rng.randn();
	a_after = 1.0 + 0.5*rng.randn();
	r0 = -2.0 + 1.0*rng.randn();
	// logU(xmin, xmax) => U(log(xmin), log(xmax))
//    changepoint = log(xmin) + (log(xmax) - log(xmin))*rng.rand();
	changepoint = log_rmin + log_dr*rng.rand();
	std::cout << "r_cp = " << exp(changepoint) << std::endl;
	r1 = 0.0 + 0.5*rng.randn();
	
	
	gp_logamp = gaussian.generate(rng);
	gp_scale = halfcauchy.generate(rng);
	gp_logalpha = uniform.generate(rng);
}

double ModelGPCP::perturb(DNest4::RNG &rng)
{
	double logH = 0.0;
	double r = rng.rand();
	
	// Perturb b_before
	if(r <= 0.1)
	{
		logH -= -0.5*pow((b_before + 2.0)/1.0, 2.0);
		b_before += 1.0*rng.randh();
		logH += -0.5*pow((b_before + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else
		{
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb a_before
	else if(r > 0.1 && r <= 0.2)
	{
		logH -= -0.5*pow((a_before - 1.0)/0.5, 2.0);
		a_before += 0.5*rng.randh();
		logH += -0.5*pow((a_before - 1.0)/0.5, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else
		{
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb a_after
	else if(r > 0.2 && r <= 0.3) {
		logH -= -0.5*pow((a_after - 1.0)/0.5, 2.0);
		a_after += 0.5*rng.randh();
		logH += -0.5*pow((a_after - 1.0)/0.5, 2.0);
		
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
	else if(r > 0.3 && r <= 0.4)
	{
		logH -= -0.5*pow((r1 + 0.0)/0.5, 2.0);
		double delta = 0.5*rng.randh();
		// FIXME: exp
//        while(r1 + delta >= exp(changepoint)) {
//            delta = 0.5*rng.randh();
//        }
		r1 += delta;
		logH += -0.5*pow((r1 + 0.0)/0.5, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else
		{
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Perturb changepoint making it larger than r1 (changepoint can't be closer to the core than r1)
	else if(r > 0.4 && r <= 0.55) {
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
	
//	// Perturb scale error
//	else if(r > 0.5 && r <= 0.6)
//	{
//		logH -= -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
//		frac_log_error_scale += 1.0 * rng.randh();
//		logH += -0.5*pow((frac_log_error_scale + 4.0) / 1.0, 2.0);
//		// Pre-reject
//		if(rng.rand() >= exp(logH))
//		{
//			return -1E300;
//		}
//		else
//		{
//			logH = 0.0;
//		}
//		// No need to re-calculate model. Just calculate loglike.
//	}
//
//	// Perturb additive error
//	else if(r > 0.6 && r <= 0.7)
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
//		else
//		{
//			logH = 0.0;
//		}
//		// No need to re-calculate model. Just calculate loglike.
//	}
	
	// Perturb r0
	else if(r > 0.55 && r <= 0.7)
	{
		
		logH -= -0.5*pow((r0 + 2.0)/1.0, 2.0);
		r0 += 1.0*rng.randh();
		logH += -0.5*pow((r0 + 2.0)/1.0, 2.0);
		
		// Pre-reject
		if(rng.rand() >= exp(logH))
		{
			return -1E300;
		}
		else
		{
			logH = 0.0;
		}
		// This shouldn't be called in case of pre-rejection
		calculate_prediction();
	}
	
	// Scale
	else if(r > 0.7 && r <= 0.8)
	{
		DNest4::Cauchy halfcauchy(0.0, 0.1);
		logH += halfcauchy.perturb(gp_scale, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	
	// Alpha
	else if(r > 0.8 && r <= 0.9)
	{
		DNest4::Uniform uniform(-5., 5.0);
		logH += uniform.perturb(gp_logalpha, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	
	else
	{
		DNest4::Gaussian gaussian(-2., 0.2);
		logH += gaussian.perturb(gp_logamp, rng);
		// No need to re-calculate model. Just calculate loglike.
	}
	
	return logH;
}

void ModelGPCP::calculate_prediction()
{
	VectorXd r = Data::get_instance().get_r();
//	mu = exp(b)*pow(r.array() + exp(r0), a);
	
	
	double r_cp = exp(changepoint);
	std::vector<Eigen::Index> before = find_less(r, r_cp);
	std::vector<Eigen::Index> after = find_ge(r, r_cp);
	
	// Continuity
	double b_after = b_before + a_before*log(r_cp + exp(r0)) - a_after*log(r_cp + r1);
	
	mu(before) = exp(b_before) * pow(r(before).array() + exp(r0), a_before);
	mu(after) = exp(b_after) * pow(r(after).array() + r1, a_after);
	
}

double ModelGPCP::log_likelihood() const
{
	double loglik = 0.0;
	VectorXd r = Data::get_instance().get_r();
	VectorXd R = Data::get_instance().get_R();
	auto n = r.size();
	VectorXd diff = R - mu;
	
//	MatrixXd C = squared_exponential_kernel(r, exp(gp_logamp), gp_scale);
	MatrixXd C = rational_quadratic_kernel(r, exp(gp_logamp), gp_scale, exp(gp_logalpha));
//	MatrixXd C_ = linear_kernel(r, 1e-05, 1.0, -exp(r0));
//	MatrixXd CC = C.array()*C_.array();

//	VectorXd disp = (mu.cwiseProduct(mu)*exp(2*frac_log_error_scale)).array() + exp(2*abs_log_error_scale);
//	C += disp.asDiagonal();
	
//	https://stackoverflow.com/a/39735211
	Eigen::LLT<Eigen::MatrixXd> llt = C.llt();
	double sqrt_det = llt.matrixL().determinant();
	// likelihood of Radius
	loglik = -0.5*n*log(2*M_PI) - log(sqrt_det) - 0.5*(llt.matrixL().solve(diff)).squaredNorm();

	return loglik;
}

void ModelGPCP::print(std::ostream &out) const
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
	out << frac_log_error_scale << "\t";
	out << gp_logamp << "\t";
	out << gp_scale << "\t";
	out << gp_logalpha;
}

std::string ModelGPCP::description() const
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
	descr += "frac_log_error_scale ";
	descr += "gp_logamp ";
	descr += "gp_scale ";
	descr += "gp_logalpha";
	
	return descr;
}
