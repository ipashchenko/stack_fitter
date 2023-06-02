#define ARMA_DONT_USE_BLAS
#include "Data.h"
#include "Model.h"
#include "ModelCP.h"
#include "ModelGP.h"
#include "utils.h"
#include "kernels.h"
#include "omp.h"

using namespace DNest4;


int main(int argc, char** argv)
{
	omp_set_num_threads(1);
	Data::get_instance().load("/home/ilya/github/stack_fitter/m87_r_fwhm.txt");
//	Sampler<Model> sampler = setup<Model>(argc, argv);
//	Sampler<ModelCP> sampler = setup<ModelCP>(argc, argv);
	Sampler<ModelGP> sampler = setup<ModelGP>(argc, argv);
	sampler.run();
	
	return 0;
}

int main0()
{
	arma::arma_config cfg;
	if( cfg.blas)
		std::cout << "BLAS enabled: "<< std::endl;
	double gp_amp = 0.1;
	double gp_scale = 1.0;
	double frac_log_error_scale = -2.;
	double abs_log_error_scale = -5.;
	
	for(int i = 0; i < 100; i++) {
		Data::get_instance().load("/home/ilya/github/stack_fitter/m87_r_fwhm.txt");
		arma::vec r = Data::get_instance().get_r();
		arma::vec R = Data::get_instance().get_R();
		arma::vec mu = R + arma::randn(r.size(), arma::distr_param(0., 0.3));
		arma::vec diff = R - mu;
		arma::vec disp = mu%mu*exp(2*frac_log_error_scale) + exp(2*abs_log_error_scale);
		
		double loglik, loglik_C;
		loglik = arma::sum(-0.5*log(2*M_PI*disp) - 0.5*(arma::pow(diff, 2.0)/disp));
		
		arma::uword n = r.size();
		
		arma::mat C = squared_exponential_kernel(r, gp_amp, gp_scale) + arma::diagmat(disp);
		arma::mat CinvX = arma::solve(C, diff);
		double logdet;
		double sign;
		bool ok = log_det(logdet, sign, C);
		if(!ok)
		{
			throw FailedDeterminantCalculationException();
		}
		// likelihood of Radius
		loglik_C = arma::sum(-0.5*n*log(2*M_PI) - 0.5*logdet - 0.5*(diff.t()*CinvX));
		
		
		std::cout << "Classic loglik = " << loglik << "\n";
		std::cout << "Covariance loglik = " << loglik_C << "\n";
	}

	
	return 0;
}