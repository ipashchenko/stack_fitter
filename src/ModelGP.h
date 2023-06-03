#ifndef MODELGP_H
#define MODELGP_H

#include <ostream>
#include "RNG.h"
#include "DNest4.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class ModelGP
{
    private:
        double a;
        double b;
        double r0;
		double frac_log_error_scale;
		double abs_log_error_scale;
		double gp_amp;
		double gp_scale;
		VectorXd mu;

        // Calculate prediction of spectra
        void calculate_prediction();

    public:
        // Constructor only gives size of params
        ModelGP();

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

#endif //MODELGP_H
