#include "Data.h"
#include <fstream>
#include <iostream>


// The static instance
Data Data::instance;

Data::Data() = default;


void Data::load(std::string filename)
{
	arma::mat A;
	A.load(filename, arma::file_type::auto_detect);
	std::cout << "Loaded data file " << filename << " with " << A.n_rows << " measurements." << "\n";
	r = A.col(0);
	R = A.col(1);
}

