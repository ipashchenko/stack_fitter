#ifndef MODELCP_H
#define MODELCP_H

#include <ostream>
#include "RNG.h"
#include "DNest4.h"
#include "armadillo"

class ModelCP
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
		arma::vec mu;
		
		// Calculate prediction of spectra
		void calculate_prediction();
	
	public:
		// Constructor only gives size of params
		ModelCP();
		
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


#endif //MODELCP_H
