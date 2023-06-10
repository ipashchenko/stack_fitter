#ifndef MODELGPCP_H
#define MODELGPCP_H

#include <ostream>
#include "RNG.h"
#include "DNest4.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class ModelGPCP
{
    private:
		double a_before;
		double a_after;
		double b_before;
		double r0;
		double r1;
		double changepoint;
		double frac_log_error_scale;
		double abs_log_error_scale;
		double log_dr;
		double log_rmin;
		double log_rmax;
		double gp_logamp;
		double gp_scale;
		double gp_logalpha;
		VectorXd mu;

        // Calculate prediction of spectra
        void calculate_prediction();

    public:
        // Constructor only gives size of params
        ModelGPCP();

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

#endif //MODELGPCP_H
