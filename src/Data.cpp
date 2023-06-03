#include "Data.h"
#include <fstream>
#include <iostream>

using namespace std;


// The static instance
Data Data::instance;

Data::Data() = default;


void Data::load(std::string filename)
{
	std::vector<double> r_;
	std::vector<double> R_;
	double x1, x2;
	
	std::fstream fin(filename, ios::in);
	while(fin>>x1 && fin>>x2)
	{
		r_.push_back(x1);
		R_.push_back(x2);
	}
	fin.close();
	
	r = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(r_.data(), r_.size());
	R = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(R_.data(), R_.size());
}

