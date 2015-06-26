#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include <omp.h>

#include "YoUtil.hpp"

#include "dnn.h"

#define DATA_TYPE float
#define T_MIN 1.0
typedef unsigned int uint;

using namespace std;




int main(int argc, char** argv){

	vector< vector <DATA_TYPE> > my_array;
	vector< vector <DATA_TYPE> > my_array_x;
	vector <DATA_TYPE>  my_array_y;
	vector< vector <DATA_TYPE> > val_x;
	vector <DATA_TYPE>  val_y;

	vector< vector <DATA_TYPE> > testing;
	vector< vector <DATA_TYPE> > test_x;
	vector <DATA_TYPE>  test_y;
	vector< vector <DATA_TYPE> > test_val_x;
	vector <DATA_TYPE>  test_val_y;


	/*
	argv[1]:  feature number
	argv[2]:  class number
	argv[3]:  hidden layer 1
	argv[4]:  hidden layer 2
	argv[5]:  epochs
	argv[6]:  learning rate
	argv[7]:  weight norms
	argv[8]:  training file name
	argv[9]:  testing file name
	*/

	srand (time(NULL));	

	read_file( argv[8], my_array, 123);
	cv_split( my_array, my_array_x, my_array_y , val_x , val_y, 0.0);

	read_file( argv[9], testing, 123);
	cv_split( testing, test_x, test_y , test_val_x , test_val_y, 0.0);

	/*
	GPU version define
	*/
	GPU gpu;

	try {
		// 1. Get context and command queues
		vector<cl::CommandQueue> cmdQueues;
		gpu.context = getContext(CL_DEVICE_TYPE_GPU, cmdQueues);
		gpu.cmdQueue = cmdQueues[0];

		// cout<<"great1"<<endl;
		
		// 2. read source code from file.
		string source = readSourceCode("kernel.cl");
		if (source.length() == 0) {
			return 255;
		}
		// cout<<"great1"<<endl;
		// 3. Compile code
		gpu.program = compile(gpu(), source);

		//              dnn( data_num, dimension, class, hidd1, hidd2 );
		dnn * net = new dnn( my_array_x.size() , atoi(argv[1]) , 
			atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

		// cout<<"great1.1"<<endl;
		
		// cout<< my_array_x.size()<<" "<< my_array_x[0].size()<<endl;
		// cout<< test_x.size()<<" "<< test_x[0].size()<<endl;
		// cout<< net->w1.size()<<" "<< net->w1[0].size()<<endl;
		// cout<<sizeof(cl_float) * my_array_x.size() * my_array_x[0].size()<<endl;
		// 4. copy data from host to device

		net->training( gpu , *gpu.program, 
			my_array_x, my_array_y, 
			test_x, test_y, 
			atoi(argv[5]), // epochs
			128, // mini batch
			atof(argv[6]), // yida
			atof(argv[7]) ); // reg

		// cout<<"great2"<<endl;


		//                                                    epochs 		 learning rate  weight norm
		// net->training(my_array_x, my_array_y, test_x, test_y, atoi(argv[5]), atof(argv[6]), atof(argv[7]));



		return 0;
	} 
	catch (cl::Error &e) {
		cerr << "\nMain: " << e.what();
		cerr << "\nError no: " << e.err() << endl;
		return 255;
	}





	return 0;
}