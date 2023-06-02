#ifndef STACK_FITTER_SRC_DATA_H
#define STACK_FITTER_SRC_DATA_H

#include "armadillo"
#include "string"

class Data {

    private:
        arma::vec r;
		arma::vec R;

    public:
        Data();
        void load(std::string filename);

        // Getters
		arma::vec get_r() const
		{
			return r;
		}
		arma::vec get_R() const
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
