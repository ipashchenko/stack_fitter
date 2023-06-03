#ifndef STACK_FITTER_SRC_DATA_H
#define STACK_FITTER_SRC_DATA_H

#include "string"
#include <iostream>
#include "fstream"
#include "vector"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;


class Data {

    private:
		VectorXd r;
		VectorXd R;

    public:
        Data();
        void load(std::string filename);

        // Getters
		VectorXd  get_r() const
		{
			return r;
		}
		VectorXd  get_R() const
		{
			return R;
		}
		
		// Singleton
    private:
        static Data instance;
    public:
        static Data& get_instance() { return instance; }

};

#endif //STACK_FITTER_SRC_DATA_H
