#ifndef MODELGPLATENT_H
#define MODELGPLATENT_H

#include <ostream>
#include "RNG.h"
#include "DNest4.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class ModelGPLatent
{
    private:
        double a;
        double b;
        double r0;
		double frac_log_error_scale;
		double abs_log_error_scale;
		double gp_logamp;
		double gp_scale;
		VectorXd v;
		MatrixXd C;
		MatrixXd L;
		VectorXd mu_gp;
		VectorXd mu_model;

		void calculate_C();
		
		void calculate_L();
		
		void calculate_gp_prediction();
		
		void calculate_model_prediction();
	
	public:
        // Constructor only gives size of params
        ModelGPLatent();

        // Generate the point from the prior
        void from_prior(DNest4::RNG& rng);

        // Metropolis-Hastings proposals
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        [[nodiscard]] double log_likelihood() const;

        // Print to stream
        void print(std::ostream& out) const;

        // Return string with column information
        [[nodiscard]] std::string description() const;
};

#endif //MODELGPLATENT_H
