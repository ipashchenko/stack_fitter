#include "Data.h"
#include "Model.h"
#include "ModelCP.h"

using namespace DNest4;


int main(int argc, char** argv)
{
	Data::get_instance().load("/home/ilya/github/stack_fitter/m87_r_fwhm.txt");
//	Sampler<Model> sampler = setup<Model>(argc, argv);
	Sampler<ModelCP> sampler = setup<ModelCP>(argc, argv);
	sampler.run();
	
	return 0;
}
