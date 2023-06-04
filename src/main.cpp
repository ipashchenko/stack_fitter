#define ARMA_DONT_USE_BLAS
#include "Data.h"
//#include "Model.h"
//#include "ModelCP.h"
#include "ModelGP.h"
#include "ModelGPCP.h"
#include "utils.h"
#include "kernels.h"
#include "omp.h"
//#include "armadillo"
//#include <Eigen/Dense>

//using Eigen::VectorXd;
//using Eigen::MatrixXd;

using namespace DNest4;


int main(int argc, char** argv)
{
//	omp_set_num_threads(1);
	Data::get_instance().load("/home/ilya/github/stack_fitter/m87_r_fwhm.txt");
//	Sampler<Model> sampler = setup<Model>(argc, argv);
//	Sampler<ModelCP> sampler = setup<ModelCP>(argc, argv);
//	Sampler<ModelGP> sampler = setup<ModelGP>(argc, argv);
	Sampler<ModelGPCP> sampler = setup<ModelGPCP>(argc, argv);
	sampler.run();

	return 0;
}

//int main()
//{
//	arma::arma_config cfg;
//	if( cfg.blas)
//		std::cout << "BLAS enabled: "<< std::endl;
//	double gp_amp = 0.1;
//	double gp_scale = 10.0;
//	double frac_log_error_scale = -2.;
//	double abs_log_error_scale = -5.;
//
//	for(int i = 0; i < 1; i++) {
//		Data::get_instance().load("/home/ilya/github/stack_fitter/m87_r_fwhm.txt");
//		VectorXd r = Data::get_instance().get_r();
//		VectorXd R = Data::get_instance().get_R();
//		std::cout << "r = " << r << "\n";
//		std::cout << "R = " << R << "\n";
//		VectorXd mu_model = R + 0.1*VectorXd::Random(r.size());
//		std::cout << "mu_model = " << mu_model << "\n";
//		VectorXd diff = R - mu_model;
//		std::cout << "diff = " << diff << "\n";
//		VectorXd disp = (mu_model.cwiseProduct(mu_model)*exp(2*frac_log_error_scale)).array() + exp(2*abs_log_error_scale);
//		std::cout << "disp = " << disp << "\n";
//		MatrixXd C = squared_exponential_kernel(r, gp_amp, gp_scale);
//		C += disp.asDiagonal();
//
//		// Convert Eigen to arma
//		arma::vec r_arma(r.data(), r.size());
//		arma::vec diff_arma(diff.data(), diff.size());
//		arma::vec disp_arma(disp.data(), disp.size());
//		arma::mat C_arma(C.data(), C.rows(), C.cols(), false, false);
//
//		double loglik_arma, loglik_eigen;
//
//		arma::uword n = r.size();
//		std::cout << "Size = " << n << "\n";
//		arma::mat CinvX = arma::solve(C_arma, diff_arma);
//		double logdet;
//		double sign;
//		bool ok = log_det(logdet, sign, C_arma);
//		std::cout << "loget arma = " << logdet << "\n";
////		std::cout << "sqrt(exp(loget)) = " << sqrt(exp(logdet)) << "\n";
//		if(!ok)
//		{
//			throw FailedDeterminantCalculationException();
//		}
//		loglik_arma = -0.5*n*log(2*M_PI) - 0.5*logdet - 0.5*arma::sum(diff_arma.t()*CinvX);
//
//		Eigen::LLT<Eigen::MatrixXd> llt = C.llt();
//		double sqrt_det = llt.matrixL().determinant();
//		std::cout << "sqrt_det = " << sqrt_det << "\n";
//		std::cout << "logdet_eigen = 2*log(sqrt_det) = " << 2.*log(sqrt_det) << "\n";
//		// likelihood of Radius
//		loglik_eigen = -0.5*n*log(2*M_PI) - log(sqrt_det) - 0.5*(llt.matrixL().solve(diff)).squaredNorm();
//
//		double factor = 0.5*(llt.matrixL().solve(r - mu_model)).squaredNorm();
//		std::cout << "factor = " << factor << '\n';
//
//		std::cout << "arma loglik = " << loglik_arma << "\n";
//		std::cout << "eigen loglik = " << loglik_eigen << "\n";
//	}
//
//
//	return 0;
//}