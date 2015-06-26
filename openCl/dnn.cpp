#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <fstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#include <stack>

#include <algorithm>

#include <random> // random generator

// #include "stopWatch.h"

#include "dnn.h"
#include "YoUtil.hpp"

#include <cassert>
#include <chrono>

using namespace std;

typedef float DATA_TYPE;

using std::string;

#define MAX_FEATURE (1024+1)

dnn::dnn( int num, int dimension, int klass, int hidd1, int hidd2  ) {
	N = num;
	dim = dimension;
	k = klass;
	h1 = hidd1;
	h2 = hidd2;
	// model['h'] = h # size of hidden layer 1
	// model['h2']= h2# size of hidden layer 2
	// model['W1']= 0.1 * np.random.randn(D,h)
	// model['b1'] = np.zeros((1,h))
	// model['W2'] = 0.1 * np.random.randn(h,h2)
	// model['b2']= np.zeros((1,h2))
	// model['W3'] = 0.1 * np.random.randn(h2,K)
	// model['b3'] = np.zeros((1,K))

	// std::random_device rd;
	// std::mt19937 gen(rd());
	// std::normal_distribution<> d(0,1.);
	// d(gen)

	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);

	std::normal_distribution<DATA_TYPE> distribution (0.0,1.0);
	

	w1.resize(dim);
	for(auto &x: w1){
		for(int i=0;i <h1 ; ++i){
			x.push_back( 0.01*distribution(generator) );
			// cout<<0.1*distribution(generator)<<" ";
		}
	}


	// cout<<0.1*distribution(generator)<<endl;

	// b1.resize(1);
	// for(auto &x: b1){
	// 	x.resize( h1 ); // zero
	// }
	b1.resize(h1);
	
	w2.resize(h1);
	for(auto &x: w2){
		for(int i=0;i <h2 ; ++i){
			x.push_back( 0.01*distribution(generator) );
		}
	}
	// b2.resize(1);
	// for(auto &x: b2){
	// 	x.resize( h2 ); 
	// }
	b2.resize(h2);

	w3.resize(h2);
	for(auto &x: w3){
		for(int i=0;i < k ; ++i){
			x.push_back( 0.01*distribution(generator) );
		}
	}
	// b3.resize(1);
	// for(auto &x: b3){
	// 	x.resize( k ); 
	// }
	b3.resize(k);



}


void read_file(char *file, 
	vector< vector <DATA_TYPE> >& my_array , 
	int max_fea){

	DATA_TYPE* features = new DATA_TYPE[MAX_FEATURE];
	
	ifstream fin;
	string istring;
	fin.open(file);

	int max_index = -INFINITY;
	int num_data=0;
	while (std::getline(fin, istring)) {
		char *cstring, *tmp;
		// int label;
		num_data++;
		// memset ( void * ptr, int value, size_t num );
		memset(features, 0, sizeof(DATA_TYPE) * MAX_FEATURE);  // replace the value of the memory size

		cstring = new char[istring.size() + 1];
		// char * strncpy ( char * destination, const char * source, size_t num );
		strncpy(cstring, istring.c_str(), istring.size()+1);

		tmp =  strtok(cstring, ": "); // split by token
		// label = atoi(tmp); // convert string to int
		tmp = strtok(NULL, ": "); // split by token

		while(tmp != NULL) {
			int id = atoi(tmp);
			if (id > max_index) max_index = id;
			tmp = strtok(NULL, ": ");
			features[id] = atof(tmp); // convert string to float
			tmp = strtok(NULL, ": ");
		}

		// cout<<features[14]<<endl;

		delete[] cstring;
	}
	delete[] features;

	// cout<< max_index<<" "<<num_data<<endl;

	fin.close();

	// DATA_TYPE* my_array = new DATA_TYPE[max_index*num_data];
	// int* label_array = new int[num_data];

	// vector< vector <DATA_TYPE> > my_array(num_data, vector<DATA_TYPE>(max_index,0));
	// vector< vector <DATA_TYPE> > my_array;
	// vector<int> label_array(num_data,0);
	// vector<int> label_array;

	// my_array(num_data, vector<DATA_TYPE>(max_index,0));
	my_array.resize(num_data);
	for(auto &x: my_array){
		// x.resize(max_index + 1); // insert label
		x.resize(max_fea +1);
	}

	// label_array.resize(num_data);

	fin.open(file);

	num_data = 0;

	while (std::getline(fin, istring)) {
		char *cstring, *tmp;
		int label;
		// memset ( void * ptr, int value, size_t num );
		// memset(my_array, 0, sizeof(DATA_TYPE) * MAX_FEATURE);  // replace the value of the memory size

		cstring = new char[istring.size() + 1];
		// char * strncpy ( char * destination, const char * source, size_t num );
		strncpy(cstring, istring.c_str(), istring.size()+1);

		tmp =  strtok(cstring, ": "); // split by token
		label = atoi(tmp); // convert string to int
		tmp = strtok(NULL, ": "); // split by token

		// label_array[num_data] = label;
		if(label==-1)
			my_array[num_data][ 0 ] = 0;
		else
			my_array[num_data][ 0 ] = label;

		while(tmp != NULL) {
			int id = atoi(tmp);
			tmp = strtok(NULL, ": ");
			// my_array[ max_index * num_data + ( id - 1) ] = atof(tmp); // convert string to float
			my_array[ num_data ][ id ] = atof(tmp); // convert string to float
			tmp = strtok(NULL, ": ");
		}

		num_data ++;

		// cout<<features[14]<<endl;

		delete[] cstring;
	}

	fin.close();

	// cout<< my_array[0]<<endl;

	// for(auto &x : my_array[0]){
	// 	cout << x<<",";
	// }
	// cout<<endl;

	// for(auto &x : label_array){
	// 	cout << x<<",";
	// }
}

void cv_split( 
	std::vector< std::vector <DATA_TYPE> > &my_array ,
	std::vector< std::vector <DATA_TYPE> > &my_array_x,
	std::vector <DATA_TYPE>  &my_array_y, 
	std::vector< std::vector <DATA_TYPE> > &val_x ,
	std::vector <DATA_TYPE>  &val_y, DATA_TYPE percent){

	int num_train = my_array.size() * (1-percent);
	int num_val = my_array.size() - num_train;
	int max_index = my_array[0].size() -1;

	// randomize array
	random_shuffle ( my_array.begin(), my_array.end() );

	my_array_x.resize(num_train);
	for(auto &x: my_array_x){
		x.resize(max_index); 
	}

	val_x.resize(num_val);
	for(auto &x: val_x){
		x.resize(max_index); 
	}

	my_array_y.resize(num_train);
	val_y.resize(num_val);

	for(size_t i=0; i< my_array.size(); ++i){
		if ((int)i < num_train)
			my_array_y[i] = my_array[i][0];
		else
			val_y[i - num_train] = my_array[i][0];

		for (int q = 0; q< max_index; ++q){
			if( (int) i< num_train)
				my_array_x[i][q] = my_array[i][q+1];
			else
				val_x[i - num_train ][q] = my_array[i][q+1];
		}
	}

}

vector< vector< vector <DATA_TYPE> > > split( vector< vector <DATA_TYPE> >& my_array, int dim, DATA_TYPE theta){
	
	vector< vector< vector <DATA_TYPE> > > sppp(2);

	for(size_t i=0 ; i< my_array.size(); ++i){
		if(my_array[i][dim] < theta ){
			sppp[0].push_back( my_array[i] );
		}
		else{
			sppp[1].push_back( my_array[i] );
		}
	}

	return sppp;
}

void dnn::training( 
	GPU &gpu,
	cl::Program &prog,
	vector< vector <DATA_TYPE> >& train_x,
	vector <DATA_TYPE> & train_y , 
	vector< vector <DATA_TYPE> >& val_x,
	vector <DATA_TYPE> & val_y , 
	int epochs, 
	int mini_batch,
	DATA_TYPE yida, 
	DATA_TYPE reg){

	// vector < vector <DATA_TYPE>>  hidden_layer_1;
	// vector < vector <DATA_TYPE>>  hidden_layer_2;
	// vector < vector <DATA_TYPE>>  scores;

	// vector < vector <DATA_TYPE>>  temp;

	// vector < vector <DATA_TYPE>>  dw3;
	// vector < vector <DATA_TYPE>>  dw2;	
	// vector < vector <DATA_TYPE>>  dw1;
	// vector <DATA_TYPE>  db3;
	// vector <DATA_TYPE>  db2;
	// vector <DATA_TYPE>  db1;

	// vector < vector <DATA_TYPE>>  dhidden2;
	// vector < vector <DATA_TYPE>>  dhidden1;


		// cl_mem buffer_b = clCreateBuffer(
	// 	context, // OpenCL context
	// 	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // Only read access from kernel,
	// 	                                         // copy data from host
	// 	sizeof(cl_double) * b.size(), // Buffer size in bytes
	// 	&b[0], // Pointer to data to copy
	// 	&errorcode); // Return code

	// clEnqueueWriteBuffer(queue, d_xx, CL_FALSE, 0, sizeof(float)*h_x.size(),
	// h_x.data(), 0, NULL, NULL);

	// cout<<"great1.2"<<endl;

	cl::Buffer test_x_(gpu(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * val_x.size() * val_x[0].size() , val_x.data() );
	cl::Buffer test_y_(gpu(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * val_y.size() , val_y.data() );

	// cout<<"great1.3"<<endl;
	/*
	Initialize weight and layer in GPU
	*/
	cl::Buffer w1_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float)* w1.size() * w1[0].size(), w1.data() );
	cl::Buffer b1_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float)*b1.size(), b1.data() );
	// cout<<"great1.4"<<endl;
	cl::Buffer w2_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * w2.size() * w2[0].size(), w2.data() );

	cl::Buffer w2_T(gpu(), CL_MEM_READ_WRITE, 
		sizeof(cl_float) * w2.size() * w2[0].size() );

	cl::Buffer b2_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * b2.size(), b2.data());
	// cout<<"great1.5"<<endl;
	cl::Buffer w3_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * w3.size() * w3[0].size(), w3.data());

	cl::Buffer w3_T(gpu(), CL_MEM_READ_WRITE, 
		sizeof(cl_float) * w3.size() * w3[0].size());

	cl::Buffer b3_(gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_float) * b3.size(), b3.data());
	// cout<<"great1.6"<<endl;
	// size: train_x.size() X h1
	cl::Buffer hidden_layer_1( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w1[0].size() );
	cl::Buffer hidden_layer_1_T( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w1[0].size() );
	// train_x.size() X h2
	cl::Buffer hidden_layer_2( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w2[0].size() );
	// train_x.size() X h2
	cl::Buffer hidden_layer_2_T( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w2[0].size() );
	// train_x.size() X class
	cl::Buffer scores( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w3[0].size() );

	// cl::Buffer temp( gpu(), CL_MEM_READ_WRITE , 
	// 	sizeof(cl_float) * );
	// cout<<"great1.7"<<endl;

	// h2 X h3
	cl::Buffer dw3( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * w2[0].size() * w3[0].size() );
	// h1 X h2
	cl::Buffer dw2( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * w1[0].size() * w3.size());
	// train_x[0].size() X h1
	cl::Buffer dw1( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * train_x[0].size() * w1[0].size() );
	// scores[0].size()
	cl::Buffer db3( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * w3[0].size() );
	// dhidden2[0].size(), h2
	cl::Buffer db2( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * w2[0].size() );
	// dhidden1[0].size(), h1
	cl::Buffer db1( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * w1[0].size() );
	// train_x.size() X h2
	cl::Buffer dhidden2( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w3.size());
	// train_x.size() X h1 
	cl::Buffer dhidden1( gpu(), CL_MEM_READ_WRITE , 
		sizeof(cl_float) * mini_batch * w2.size());

	// mini batch example
	int num_examples = (int)train_x.size()/mini_batch;
	
	// cout<<"great1.3"<<endl;

	stopWatch timer;

	for(int iter=0; iter< epochs; ++iter){
		DATA_TYPE t1;
		
		timer.start();

		vector< DATA_TYPE > data_loss (1,0);
		cl::Buffer data_loss22 (gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
			sizeof(cl_float) * data_loss.size() , data_loss.data() );

		for(int batch = 0; batch< num_examples; batch++ ){
			// cout<< batch <<endl;
			vector< vector <DATA_TYPE> > temp_x ( mini_batch , vector <DATA_TYPE>(train_x[0].size(),0 ) );
			vector <DATA_TYPE> temp_y ( mini_batch , 0 );
			for(int index= batch * mini_batch ; index < (batch +1)* mini_batch; index++ ){
				for(size_t dim=0; dim< train_x[0].size(); dim++){
					temp_x[index - batch * mini_batch ][dim] = train_x[index][dim];
				}
				temp_y[index - batch * mini_batch] = train_y[index];
			}
			// cout<<"done"<<endl;
			// need to make mini batch inside train !!!!!!
			cl::Buffer my_array_x_(gpu(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				sizeof(cl_float) * temp_x.size() * temp_x[0].size() , temp_x.data() );
			cl::Buffer my_array_x_T (gpu(), CL_MEM_READ_WRITE, 
				sizeof(cl_float) * temp_x.size() * temp_x[0].size() );

			cl::Buffer my_array_y_(gpu(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				sizeof(cl_float) * temp_y.size() , temp_y.data() );

			


			/*
			feed forward propagation
			*/

			// size: train_x.size() X h1
			// hidden_layer_1 = ( matrix_matrix(train_x, w1, false, false) );
			matrix_matrix( gpu, prog, (int)temp_x.size(), 
				(int)temp_x[0].size(), (int)w1[0].size(), 
				my_array_x_, w1_ , hidden_layer_1 );

			// m_v_add(hidden_layer_1, b1);
			// hiddenlayer 1:  mini_batch * w1[0].size()
			m_v_add( gpu, prog, mini_batch, (int)w1[0].size(), hidden_layer_1, b1_ );

			// sigmoid(hidden_layer_1);
			sigmoid( gpu, prog, mini_batch, (int)w1[0].size(), hidden_layer_1 );

			// // w2: h1 X h2
			// // so: train_x.size() X h2
			// hiddenlayer 1:  mini_batch * w1[0].size()
			// hidden_layer_2 = matrix_matrix(hidden_layer_1, w2, false, false);
			matrix_matrix( gpu, prog, (int)mini_batch, 
				(int)w1[0].size() , (int)w2[0].size(), 
				hidden_layer_1, w2_ , hidden_layer_2 );

			// m_v_add(hidden_layer_2, b2);
			m_v_add( gpu, prog, mini_batch, (int)w2[0].size(), hidden_layer_2, b2_ );

			// sigmoid(hidden_layer_2);
			sigmoid( gpu, prog, mini_batch, (int)w2[0].size(), hidden_layer_2 );

			// hiddenlayer 2:  mini_batch * w2[0].size()
			// // train_x.size() X class
			// scores = matrix_matrix(hidden_layer_2, w3, false, false);
			matrix_matrix( gpu, prog, (int)mini_batch, 
				(int)w2[0].size() , (int)w3[0].size(), 
				hidden_layer_2, w3_ , scores );

			// m_v_add(scores, b3);
			m_v_add( gpu, prog, mini_batch, (int)w3[0].size(), scores, b3_ );

			// // exp_scores = np.exp(scores)
			// // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
			// /*
			// compute probability

			// softmax
			// #     e = T.exp(X)
			// #     return e / T.sum(e, axis=1).dimshuffle(0, 'x')

	  //       */

			// softmax( gpu, prog, mini_batch, (int)w3[0].size(), scores );

			// for (size_t i=0 ; i<  scores.size();++i){
			// 	DATA_TYPE temp = 0;
			// 	for(size_t j=0; j< scores[0].size(); ++j){ // class num
			// 		temp += exp(scores[i][j]); // normalize term
			// 	}
			// 	for (size_t j=0; j< scores[0].size(); ++j){
			// 		scores[i][j] = exp(scores[i][j]) / temp; // transform to probalitity
			// 	}
			// }

			// // # compute the loss: average cross-entropy loss and regularization
			// // corect_logprobs = -np.log(probs[range(num_examples),y])
			// // data_loss = np.sum(corect_logprobs)/num_examples
			// // reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)+ 0.5*reg*np.sum(W3*W3)
			// // loss = data_loss + reg_loss

			// /*
			// compute cost function
			// */
			// // z      = [ 0.34, 0.66 ]
			// // target = [ 1 , 0 ]
			// // def multinominal_cross_entropy(z, target):
			// //     loss = - T.mean( target * T.log(z) + (1 - target) * T.log(1 - z))
			// // multinominal_cross_entropy(p_y_given_x, target) 

			// DATA_TYPE data_loss [1] = {0.};
			// vector< DATA_TYPE > data_loss (1,0);

			// for(size_t i=0; i<scores.size();++i){
			// 	data_loss += log(scores[i][train_y[i]]) ;
			// }
			// data_loss = (-1.)*data_loss / mini_batch;

			// cl::Buffer data_loss22 (gpu(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
			// 	sizeof(cl_float) * data_loss.size() , data_loss.data() );

			softmax( gpu, prog, mini_batch, (int)w3[0].size(), scores );

			multinominal_cross_entropy( gpu, prog, 
				(int)mini_batch, (int)w3[0].size() , scores, my_array_y_ , data_loss22 );

			// cout<<"great 2"<<endl;
			// DATA_TYPE reg_loss = 0;

			// temp = matrix_matrix(w1,w1,true,false);
			// reg_loss += 0.5*reg*sum(temp);

			// temp = matrix_matrix(w2,w2,true,false);
			// reg_loss += 0.5*reg*sum(temp);

			// temp = matrix_matrix(w3,w3,true,false);
			// reg_loss += 0.5*reg*sum(temp);

			// data_loss = data_loss + reg_loss;

			// // # compute the gradient on scores
			// for(size_t i=0;i<scores.size(); ++i){
			// 	scores[i][ train_y[i] ] -= 1;
			// 	for(size_t q=0;q<scores[0].size(); ++q)
			// 		scores[i][q] /= num_examples;
			// }
			gradient_score( gpu, prog, 
				(int)mini_batch, (int)w3[0].size() , 
				scores, my_array_y_  );

			// cout<<"great 2.1"<<endl;

			// # BACKPROP HERE
			// dw3 = (hidden_layer_2.T).dot(dscores)
			// db3 = np.sum(dscores, axis=0, keepdims=True)
			// hidden2: train_x.size() X h2
			//  T: h2 X train_x.size()
			// score: train_x.size() X class
			// so: h2 X class
			// dw3 = matrix_matrix(hidden_layer_2, scores, true,false);

			transpose( gpu, prog, 
				(int)mini_batch, (int)w2[0].size() ,
				hidden_layer_2 , hidden_layer_2_T );

			matrix_matrix( gpu, prog, 
				(int)w2[0].size() , (int)mini_batch, // transpose of hidden layer 2
				(int)w3[0].size(), 
				hidden_layer_2_T, scores , dw3 );

			// for(size_t i=0; i<scores[0].size(); ++i){
			// 	db3.push_back(0);
			// 	for(size_t q=0; q< scores.size(); ++q){
			// 		db3[i] += scores[q][i];
			// 	}
			// }
			bias_sum( gpu, prog,
				(int)w3[0].size(), (int)mini_batch,
				scores, db3 );

			// cout<<"great 2.2"<<endl;

			// #backprop sigmoid nonlinearity here
			// dhidden2 = dscores.dot(w3.T)*sigmoid_grad(hidden_layer_2)
			// score: train_x.size() X class
			// w3: h2 X class
			// T: class X h2
			// so : train_x.size() X h2
			// dhidden2 = matrix_matrix( scores, w3, false, true );

			transpose( gpu, prog, 
				(int)w2[0].size(), (int)w3[0].size() ,
				w3_ , w3_T );

			matrix_matrix( gpu, prog, 
				(int)mini_batch, (int)w3[0].size() , 
				(int)w2[0].size(), // transpose of w3
				scores, w3_T , dhidden2 );


			// sigmoid_grad(hidden_layer_2);
			// hidden2: train_x.size() X h2
			sigmoid_grad( gpu, prog, 
				(int)mini_batch, (int)w2[0].size() , 
				hidden_layer_2 );

			// cout<<"great 2.3"<<endl;
			// for(size_t i=0; i< dhidden2.size(); ++i){
			// 	for(size_t q =0; q< dhidden2[0].size(); ++q){
			// 		dhidden2[i][q] *= hidden_layer_2[i][q];
			// 	}
			// }
			// dhidden2: train_x.size() X h2
			ele_multiply(gpu, prog, 
				(int)mini_batch, (int)w2[0].size() , 
				dhidden2, hidden_layer_2 );
			// cout<<"great 2.4"<<endl;
			// dw2 = (hidden_layer.T).dot(dhidden2)
			// db2 = np.sum(dhidden2, axis=0)
			// hidden1 size: train_x.size() X h1
			// T : h1 X train_x.size()
			// dhidden2 : train_x.size() X h2
			// so: h1 X h2
			// dw2 = matrix_matrix(hidden_layer_1 , dhidden2, true , false);
			transpose( gpu, prog, 
				(int)mini_batch, (int)w1[0].size() ,
				hidden_layer_1 , hidden_layer_1_T );

			matrix_matrix( gpu, prog, 
				(int)w1[0].size() , (int)mini_batch , // hidden layer 1 transpose
				(int)w2[0].size(), 
				hidden_layer_1_T, dhidden2 , dw2 );

			// cout<<"great 2.4"<<endl;
			// for(size_t i=0; i<dhidden2[0].size(); ++i){
			// 	db2.push_back(0);
			// 	for(size_t q=0; q< dhidden2.size(); ++q){
			// 		db2[i] += dhidden2[q][i];
			// 	}
			// }
			bias_sum( gpu, prog,
				(int)w2[0].size(), (int)mini_batch,
				dhidden2, db2 );

			// dhidden = dhidden2.dot(w2.T)*sigmoid_grad(hidden_layer_1)
			// dhidden2: train_x.size() X h2
			// w2: h1 X h2
			// T: h2 X h1
			// so: train_x.size() X h1

			// dhidden1 = matrix_matrix(dhidden2, w2, false, true);
			transpose( gpu, prog, 
				(int)w1[0].size(), (int)w2[0].size() ,
				w2_ , w2_T );

			matrix_matrix( gpu, prog, 
				(int)mini_batch , (int)w2[0].size() , 
				(int)w1[0].size(), // w2 transpose
				dhidden2, w2_T , dhidden1 );

			// sigmoid_grad(hidden_layer_1);
			// hidden1 size: train_x.size() X h1
			sigmoid_grad( gpu, prog, 
				(int)mini_batch, (int)w1[0].size() , 
				hidden_layer_1 );
			// cout<<"great 2.5"<<endl;
			// for(size_t i=0; i< dhidden1.size(); ++i){
			// 	for(size_t q =0; q< dhidden1[0].size(); ++q){
			// 		dhidden1[i][q] *= hidden_layer_1[i][q];
			// 	}
			// }
			ele_multiply(gpu, prog, 
				(int)mini_batch, (int)w1[0].size() , 
				dhidden1, hidden_layer_1 );

			// dw1 =  np.dot(X.T, dhidden)
			// db1 = np.sum(dhidden, axis=0)
			// train_x: train_x.size() X train_x[0].size()
			// T : train_x[0].size() X train_x.size()
			// dhidden1: train_x.size() X h1
			// so: train_x[0].size() X h1
			// dw1 = matrix_matrix(train_x, dhidden1, true, false);
			transpose( gpu, prog, 
				(int)mini_batch, (int) train_x[0].size(),
				my_array_x_ , my_array_x_T );

			matrix_matrix( gpu, prog, 
				(int) train_x[0].size(), (int)mini_batch , // train_x transpose
				(int)w1[0].size(), 
				my_array_x_T, dhidden1 , dw1 );

			// cout<<"great 2.6"<<endl;
			// for(size_t i=0; i<dhidden1[0].size(); ++i){
			// 	db1.push_back(0);
			// 	for(size_t q=0; q< dhidden1.size(); ++q){
			// 		db1[i] += dhidden1[q][i];
			// 	}
			// }
			bias_sum( gpu, prog,
				(int)w1[0].size(), (int)mini_batch,
				dhidden1, db1 );

			// # add regularization
			// dW3+= reg * W3
			// dW2 += reg * W2
			// dW1 += reg * W1
			// for(size_t i=0; i<dw3.size(); ++i)
			// 	for (size_t q=0; q<dw3[0].size(); ++q)
			// 		dw3[i][q] += reg * w3[i][q];
			// for(size_t i=0; i<dw2.size(); ++i)
			// 	for (size_t q=0; q<dw2[0].size(); ++q)
			// 		dw2[i][q] += reg * w2[i][q];
			// for(size_t i=0; i<dw1.size(); ++i)
			// 	for (size_t q=0; q<dw1[0].size(); ++q)
			// 		dw1[i][q] += reg * w1[i][q];

			// # update
			// W1 += -step_size * dW1
			// b1 += -step_size * db1
			// W2 += -step_size * dW2
			// b2 += -step_size * db2
			// W3 += -step_size * dW3
			// b3 += -step_size * db3

			// for(auto &x: w1){
			// 	for (auto &y: x){
			// 		cout<< y<<" ";
			// 	}
			// 	cout<<endl;
			// }
			// cout<<endl<<endl<<endl;
			// cout<<endl<<endl<<endl;
			// cout<<endl<<endl<<endl;
			// cout<<"************************"<<endl;
			// cout<<endl<<endl<<endl;

			// update_2(w1, dw1, yida);
			// update_1(b1, db1, yida);
			// update_2(w2, dw2, yida);
			// update_1(b2, db2, yida);
			// update_2(w3, dw3, yida);
			// update_1(b3, db3, yida);
			// cout<<"great 2.7"<<endl;
			update2(gpu, prog,
				(int)w1.size(), (int)w1[0].size(), yida,
				w1_, dw1 );
			update1(gpu, prog,
				(int)b1.size(), yida,
				b1_, db1 );
			update2(gpu, prog,
				(int)w2.size(), (int)w2[0].size(), yida,
				w2_, dw2 );
			update1(gpu, prog,
				(int)b2.size(), yida,
				b2_, db2 );
			update2(gpu, prog,
				(int)w3.size(), (int)w3[0].size(), yida,
				w3_, dw3 );
			update1(gpu, prog,
				(int)b3.size(), yida,
				b3_, db3 );


		}


		// cout<<yida<<endl;

		timer.stop();
		t1 = timer.elapsedTime();
		cout<<t1<<endl;
		// if (iter%1==0){
		// 	// printf("epochs %d: loss %f, val: %f, t: %f \n", 
		// 	// 	iter, data_loss , predict(val_x, val_y), t1 );
		// 	printf("epochs %d: , t: %f \n", 
		// 		iter , t1 );
		// }	

	}

}

void dnn::training( 
	vector< vector <DATA_TYPE> >& train_x,
	vector <DATA_TYPE> & train_y , 
	vector< vector <DATA_TYPE> >& val_x,
	vector <DATA_TYPE> & val_y , 
	int epochs, DATA_TYPE yida, DATA_TYPE reg){

	vector < vector <DATA_TYPE>>  hidden_layer_1;
	vector < vector <DATA_TYPE>>  hidden_layer_2;
	vector < vector <DATA_TYPE>>  scores;

	vector < vector <DATA_TYPE>>  temp;

	vector < vector <DATA_TYPE>>  dw3;
	vector < vector <DATA_TYPE>>  dw2;	
	vector < vector <DATA_TYPE>>  dw1;
	vector <DATA_TYPE>  db3;
	vector <DATA_TYPE>  db2;
	vector <DATA_TYPE>  db1;

	vector < vector <DATA_TYPE>>  dhidden2;
	vector < vector <DATA_TYPE>>  dhidden1;

	DATA_TYPE num_examples = (DATA_TYPE)train_x.size();

	stopWatch timer;

	for(int iter=0; iter< epochs; ++iter){
		DATA_TYPE t1;
		
		timer.start();
		/*
		feed forward propagation
		*/		
		hidden_layer_1 = ( matrix_matrix(train_x, w1, false, false) );
		m_v_add(hidden_layer_1, b1);
		sigmoid(hidden_layer_1);

		hidden_layer_2 = matrix_matrix(hidden_layer_1, w2, false, false);

		m_v_add(hidden_layer_2, b2);
		sigmoid(hidden_layer_2);

		scores = matrix_matrix(hidden_layer_2, w3, false, false);
		m_v_add(scores, b3);

		// exp_scores = np.exp(scores)
		// probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
		/*
		compute probability

		softmax
		#     e = T.exp(X)
		#     return e / T.sum(e, axis=1).dimshuffle(0, 'x')

        */
		for (size_t i=0 ; i<  scores.size();++i){
			DATA_TYPE temp = 0;
			for(size_t j=0; j< scores[0].size(); ++j){ // class num
				temp += exp(scores[i][j]); // normalize term
			}
			for (size_t j=0; j< scores[0].size(); ++j){
				scores[i][j] = exp(scores[i][j]) / temp; // transform to probalitity
			}
		}

		// # compute the loss: average cross-entropy loss and regularization
		// corect_logprobs = -np.log(probs[range(num_examples),y])
		// data_loss = np.sum(corect_logprobs)/num_examples
		// reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)+ 0.5*reg*np.sum(W3*W3)
		// loss = data_loss + reg_loss

		/*
		compute cost function
		*/
		// z      = [ 0.34, 0.66 ]
		// target = [ 1 , 0 ]
		// def multinominal_cross_entropy(z, target):
		//     loss = - T.mean( target * T.log(z) + (1 - target) * T.log(1 - z))
		// multinominal_cross_entropy(p_y_given_x, target) 

		DATA_TYPE data_loss = 0;
		for(size_t i=0; i<scores.size();++i){
			data_loss += log(scores[i][train_y[i]]) ;
		}
		data_loss = (-1)*data_loss / num_examples;
		
		DATA_TYPE reg_loss = 0;

		temp = matrix_matrix(w1,w1,true,false);
		reg_loss += 0.5*reg*sum(temp);

		temp = matrix_matrix(w2,w2,true,false);
		reg_loss += 0.5*reg*sum(temp);

		temp = matrix_matrix(w3,w3,true,false);
		reg_loss += 0.5*reg*sum(temp);

		data_loss = data_loss + reg_loss;

		// # compute the gradient on scores
		for(size_t i=0;i<scores.size(); ++i){
			scores[i][ train_y[i] ] -= 1;
			for(size_t q=0;q<scores[0].size(); ++q)
				scores[i][q] /= num_examples;
		}

		// # BACKPROP HERE
		// dw3 = (hidden_layer_2.T).dot(dscores)
		// db3 = np.sum(dscores, axis=0, keepdims=True)

		dw3 = matrix_matrix(hidden_layer_2, scores, true,false);

		for(size_t i=0; i<scores[0].size(); ++i){
			db3.push_back(0);
			for(size_t q=0; q< scores.size(); ++q){
				db3[i] += scores[q][i];
			}
		}

		// #backprop sigmoid nonlinearity here
		// dhidden2 = dscores.dot(w3.T)*sigmoid_grad(hidden_layer_2)

		dhidden2 = matrix_matrix( scores, w3, false, true );

		sigmoid_grad(hidden_layer_2);

		for(size_t i=0; i< dhidden2.size(); ++i){
			for(size_t q =0; q< dhidden2[0].size(); ++q){
				dhidden2[i][q] *= hidden_layer_2[i][q];
			}
		}

		// dw2 = (hidden_layer.T).dot(dhidden2)
		// db2 = np.sum(dhidden2, axis=0)
		dw2 = matrix_matrix(hidden_layer_1 , dhidden2, true , false);
		for(size_t i=0; i<dhidden2[0].size(); ++i){
			db2.push_back(0);
			for(size_t q=0; q< dhidden2.size(); ++q){
				db2[i] += dhidden2[q][i];
			}
		}

		// dhidden = dhidden2.dot(w2.T)*sigmoid_grad(hidden_layer_1)
		dhidden1 = matrix_matrix(dhidden2, w2, false, true);
		sigmoid_grad(hidden_layer_1);
		for(size_t i=0; i< dhidden1.size(); ++i){
			for(size_t q =0; q< dhidden1[0].size(); ++q){
				dhidden1[i][q] *= hidden_layer_1[i][q];
			}
		}

		// dw1 =  np.dot(X.T, dhidden)
		// db1 = np.sum(dhidden, axis=0)
		dw1 = matrix_matrix(train_x, dhidden1, true, false);

		for(size_t i=0; i<dhidden1[0].size(); ++i){
			db1.push_back(0);
			for(size_t q=0; q< dhidden1.size(); ++q){
				db1[i] += dhidden1[q][i];
			}
		}

		// # add regularization
		// dW3+= reg * W3
		// dW2 += reg * W2
		// dW1 += reg * W1
		for(size_t i=0; i<dw3.size(); ++i)
			for (size_t q=0; q<dw3[0].size(); ++q)
				dw3[i][q] += reg * w3[i][q];
		for(size_t i=0; i<dw2.size(); ++i)
			for (size_t q=0; q<dw2[0].size(); ++q)
				dw2[i][q] += reg * w2[i][q];
		for(size_t i=0; i<dw1.size(); ++i)
			for (size_t q=0; q<dw1[0].size(); ++q)
				dw1[i][q] += reg * w1[i][q];

		// # update
		// W1 += -step_size * dW1
		// b1 += -step_size * db1
		// W2 += -step_size * dW2
		// b2 += -step_size * db2
		// W3 += -step_size * dW3
		// b3 += -step_size * db3

		// for(auto &x: w1){
		// 	for (auto &y: x){
		// 		cout<< y<<" ";
		// 	}
		// 	cout<<endl;
		// }
		// cout<<endl<<endl<<endl;
		// cout<<endl<<endl<<endl;
		// cout<<endl<<endl<<endl;
		// cout<<"************************"<<endl;
		// cout<<endl<<endl<<endl;
		update_2(w1, dw1, yida);
		update_1(b1, db1, yida);
		update_2(w2, dw2, yida);
		update_1(b2, db2, yida);
		update_2(w3, dw3, yida);
		update_1(b3, db3, yida);

		// cout<<yida<<endl;

		timer.stop();
		t1 = timer.elapsedTime();
		// cout<<t1<<endl;
		if (iter%1==0){
			printf("epochs %d: loss %f, val: %f, t: %f \n", 
				iter, data_loss , predict(val_x, val_y), t1 );
		}	

	}




}

DATA_TYPE dnn::predict( std::vector< std::vector <DATA_TYPE> >  & train_x,
		std::vector <DATA_TYPE>   & train_y ){
	// # evaluate training set accuracy
	// if NONLINEARITY == 'RELU':
	//     hidden_layer = relu(np.dot(X, W1) + b1)
	//     hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
	// elif NONLINEARITY == 'SIGM':
	//     hidden_layer = sigmoid(np.dot(X, W1) + b1)
	//     hidden_layer2 = sigmoid(np.dot(hidden_layer, W2) + b2)
	// scores = np.dot(hidden_layer2, W3) + b3
	// predicted_class = np.argmax(scores, axis=1)
	// print 'training accuracy: %.2f' % (np.mean(predicted_class == y))  

	vector < vector <DATA_TYPE>>  hidden_layer_1;
	vector < vector <DATA_TYPE>>  hidden_layer_2;
	vector < vector <DATA_TYPE>>  scores;

	vector <int>  predict_class;
	
	hidden_layer_1 = ( matrix_matrix(train_x, w1, false, false) );

	m_v_add(hidden_layer_1, b1);
	sigmoid(hidden_layer_1);

	hidden_layer_2 = matrix_matrix(hidden_layer_1, w2, false, false);
	m_v_add(hidden_layer_2, b2);
	sigmoid(hidden_layer_2);

	scores = matrix_matrix(hidden_layer_2, w3, false, false);
	m_v_add(scores, b3);
	
	for(size_t i=0; i< scores.size(); ++i){
		predict_class.push_back(0);
		DATA_TYPE temp = -INFINITY;
		for (size_t q=0; q<scores[0].size(); ++q){
			if (scores[i][q]>temp){
				temp = scores[i][q];
				predict_class[i] = q; // y label
			}
		}
	}

	DATA_TYPE correct =0;
	for (size_t i=0; i<predict_class.size(); ++i){
		if (predict_class[i] == train_y[i])
			correct++;
	}

	return correct/(DATA_TYPE)predict_class.size();

}


/* 

	openCL Kernel operation

*/
void matrix_matrix(
	GPU &gpu,
	cl::Program &prog,
	const int  M,
	const int  K,
	const int  N,
	cl::Buffer &my_array_x_,
	cl::Buffer &w1,
	cl::Buffer &hidden_layer_1
	){

	cl::Kernel kernel( prog, "matrix_matrix");

	// size: train_x.size() X h1
	// hidden_layer_1 = ( matrix_matrix(train_x, w1, false, false) );

	kernel.setArg(0, (cl_int) M );
	kernel.setArg(1, (cl_int) N );
	kernel.setArg(2, (cl_int) K );
	kernel.setArg(3, my_array_x_);
	kernel.setArg(4, w1);
	kernel.setArg(5, hidden_layer_1);

	size_t lSize=16;
	cl::NDRange local( lSize, lSize );
	cl::NDRange global( ((cl_int) M + lSize-1) / 
		lSize * lSize, ((cl_int) N + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();

}

void m_v_add(
	GPU &gpu,
	cl::Program &prog,
	const int  mini_batch,
	const int  K,
	cl::Buffer &hidden_layer_1,
	cl::Buffer &b1
	){

	// m_v_add(hidden_layer_1, b1);
	// hiddenlayer 1:  mini_batch * w1[0].size()
	cl::Kernel kernel( prog, "m_v_add");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) K );
	kernel.setArg(2, hidden_layer_1 );
	kernel.setArg(3,  b1 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();
}

void sigmoid( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &hidden_layer_1){

	cl::Kernel kernel( prog, "sigmoid");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) K );
	kernel.setArg(2, hidden_layer_1 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();


}

void softmax( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &scores){

	cl::Kernel kernel( prog, "softmax");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) K );
	kernel.setArg(2, scores );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();

}

void multinominal_cross_entropy(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int class_num,
	cl::Buffer &scores,
	cl::Buffer &temp_y,
	cl::Buffer &data_loss
	){

	cl::Kernel kernel( prog, "multinominal_cross_entropy");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) class_num );
	kernel.setArg(2, scores );
	kernel.setArg(3, temp_y );
	kernel.setArg(4, data_loss );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();


}


void gradient_score(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int class_num,
	cl::Buffer &scores,
	cl::Buffer &temp_y
	){

	cl::Kernel kernel( prog, "gradient_score");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) class_num );
	kernel.setArg(2, scores );
	kernel.setArg(3, temp_y );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();
}

void transpose( 
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch, 
	const int h2,
	cl::Buffer & hidden_layer_2 , 
	cl::Buffer & hidden_layer_2_T ){

	cl::Kernel kernel( prog, "transpose");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) h2 );
	kernel.setArg(2, hidden_layer_2 );
	kernel.setArg(3, hidden_layer_2_T );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();

}

void bias_sum( 
	GPU &gpu,
	cl::Program &prog, 
	const int class_num, 
	const int mini_batch,
	cl::Buffer & scores, 
	cl::Buffer & bias ){

	cl::Kernel kernel( prog, "bias_sum");
	kernel.setArg(0, (cl_int) class_num );
	kernel.setArg(1, (cl_int) mini_batch );
	kernel.setArg(2, scores );
	kernel.setArg(3, bias );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)class_num + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();



}
	
void sigmoid_grad( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &hidden_layer_1){

	cl::Kernel kernel( prog, "sigmoid_grad");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) K );
	kernel.setArg(2, hidden_layer_1 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();

}

void ele_multiply(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int K,
	cl::Buffer & array1, 
	cl::Buffer & array2 ){

	cl::Kernel kernel( prog, "ele_multiply");
	kernel.setArg(0, (cl_int) mini_batch );
	kernel.setArg(1, (cl_int) K );
	kernel.setArg(2, array1 );
	kernel.setArg(3, array2 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)mini_batch + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();



}


void update2(
	GPU &gpu,
	cl::Program &prog, 
	const int array1_x,
	const int array1_y,
	const float yida,
	cl::Buffer & array1, 
	cl::Buffer & array2 ){

	cl::Kernel kernel( prog, "update_2");
	kernel.setArg(0, (cl_int) array1_x );
	kernel.setArg(1, (cl_int) array1_y );
	kernel.setArg(2, (cl_float) yida );
	kernel.setArg(3, array1 );
	kernel.setArg(4, array2 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)array1_x + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();

}



void update1(
	GPU &gpu,
	cl::Program &prog, 
	const int array1_x,
	const float yida,
	cl::Buffer & array1, 
	cl::Buffer & array2 ){

	cl::Kernel kernel( prog, "update_1");
	kernel.setArg(0, (cl_int) array1_x );
	kernel.setArg(1, (cl_float) yida );
	kernel.setArg(2, array1 );
	kernel.setArg(3, array2 );

	size_t lSize= 256 ;
	cl::NDRange local( lSize );
	cl::NDRange global( ( (cl_int)array1_x + lSize-1) / lSize * lSize );

	cl::Event event;
	gpu.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
	event.wait();
}


/*

	Normal training operation

*/


void sigmoid (std::vector< std::vector <DATA_TYPE> >& array){
	// x = 1/(1+np.exp(-x))
	for (size_t x=0; x < array.size(); ++x){
		for(size_t y = 0; y<array[0].size(); ++y){
			array[x][y] = 1 / ( 1 + exp(-1*array[x][y]) );
		}
	}
}

void sigmoid_grad(std::vector< std::vector <DATA_TYPE> >& array){
	// (x)*(1-x)
	for (size_t x=0; x < array.size(); ++x){
		for(size_t y = 0; y<array[0].size(); ++y){
			array[x][y] = array[x][y] * ( 1- array[x][y] );
		}
	}
}

DATA_TYPE sum(std::vector< std::vector <DATA_TYPE> >& array1){
	DATA_TYPE sum=0;
	for (size_t i=0;i<array1.size(); ++i){
		for(size_t q=0; q< array1[0].size(); ++q){
			sum += array1[i][q];
		}
	}
	return sum;
}

void update_1 ( std::vector <DATA_TYPE> & array1,
	std::vector <DATA_TYPE> & array2,
	DATA_TYPE b ){

	for (size_t i=0; i< array1.size(); ++i){
			array1[i] += -b* array2[i] ;
	}
}

void update_2 (std::vector< std::vector <DATA_TYPE> >& array1,
	std::vector< std::vector <DATA_TYPE> >& array2,
	DATA_TYPE b ){
	for (size_t i=0; i< array1.size(); ++i){
		for(size_t q =0 ;q < array1[0].size(); ++q){
			array1[i][q] += -b * array2[i][q] ;
		}
	}
}



void m_v_add (std::vector< std::vector <DATA_TYPE> >& array,
	std::vector <DATA_TYPE> & b ){

	assert(array[0].size()==b.size());
	// cout<<"==="<<endl;
	for (size_t i=0; i< array.size(); ++i){ // 20
		for(size_t q =0 ;q < array[0].size(); ++q){ //100
			array[i][q] += b[q];
		}
	}
}

std::vector< std::vector <DATA_TYPE> > matrix_matrix(std::vector< std::vector <DATA_TYPE> >& array1, 
	std::vector< std::vector <DATA_TYPE> >& array2, bool transpose_1, bool transpose_2){

	size_t x = array1.size();
	size_t y = array2[0].size();

	if (transpose_1){

		size_t z = array1[0].size();
		vector< vector <DATA_TYPE> > array1_T(z, vector<DATA_TYPE> (x,0));
		for(size_t i=0; i<x; ++i ){
			for(size_t j=0; j<z; ++j){
				array1_T[j][i] = array1[i][j];
			}
		}

		// array1 transpose
		assert(array1.size()==array2.size());

		// cout<<x<<" "<<array1[0].size()<<" "<<array2.size()<<" "<<y<<endl;
		vector< vector <DATA_TYPE> > result(z, vector<DATA_TYPE> (y,0));
		for (size_t i=0; i < array1_T.size(); ++i){
			for(size_t k = 0; k<array2[0].size(); ++k){
				for (size_t j=0; j< array1_T[0].size(); ++j){
					result[i][k] += array1_T[i][j] * array2[j][k];
				}
			}
		}

		array1_T.clear();
		return result;
	}
	else if (transpose_2){
		size_t z = array2.size();
		vector< vector <DATA_TYPE> > array2_T(y, vector<DATA_TYPE> (z,0));
		for(size_t i=0; i<z; ++i ){
			for(size_t j=0; j<y; ++j){
				array2_T[j][i] = array2[i][j];
			}
		}

		// array 2 transpose
		assert(array1[0].size()==array2[0].size());
		// cout<<x<<" "<<array1[0].size()<<" "<<array2.size()<<" "<<y<<endl;
		vector< vector <DATA_TYPE> > result(x, vector<DATA_TYPE> (z,0));
		for (size_t i=0; i < array1.size(); ++i){
			for(size_t k = 0; k<array2_T[0].size(); ++k){
				for (size_t j=0; j< array1[0].size(); ++j){
					result[i][k] += array1[i][j] * array2_T[j][k];
				}
			}
		}

		array2_T.clear();
		return result;
	}
	else{

		assert(array1[0].size()==array2.size());

		vector< vector <DATA_TYPE> > result(x, vector<DATA_TYPE> (y,0));
		for (size_t i=0; i < array1.size(); ++i){
			for(size_t k = 0; k<array2[0].size(); ++k){
				for (size_t j=0; j< array1[0].size(); ++j){
					result[i][k] += array1[i][j] * array2[j][k];
				}
			}
		}
		return result;
	}

	
	// cout<<result.size()<<" "<<result[0].size()<<endl;
	
}

