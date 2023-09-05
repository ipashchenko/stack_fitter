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
	
	std::vector<double> r__, R__;
	int size = r_.size();
	for(int i=0; i < size; i += 1)
	{
		if(r_[i] < 30.1) {
			r__.push_back(r_[i]);
			R__.push_back(R_[i]);
		}
	}
	
	r = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(r__.data(), r__.size());
	R = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(R__.data(), R__.size());
	// This makes first r 1E-512
//	r = r(Eigen::seq(0, Eigen::last, 2));
//	R = R(Eigen::seq(0, Eigen::last, 2));
}

