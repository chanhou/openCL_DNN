#ifndef DNN
#define DNN

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

typedef float DATA_TYPE;
// #define DATA_TYPE float

#include <vector>
#include <cstddef>
#include <stdio.h>
#include <stack>

/*
reference:
1. http://cs224d.stanford.edu/notebooks/vanishing_grad_example.html
*/
/*
	GPU version define
*/
struct GPU {
	cl::CommandQueue cmdQueue;
	cl::Program *program;
	cl::Context *context;
	GPU() {
		program = nullptr;
		context = nullptr;
	}
	~GPU() {
		delete program;
		delete context;
	}
	cl::Context& operator() () {
		return *context;
	}
};

class dnn {
public:
	dnn( int num, int dimension, int klass, int hidd1, int hidd2 );
	~dnn();

	void print_file( FILE * abc ) const;

	void training(
		std::vector< std::vector <DATA_TYPE> >  & train_x,
		std::vector <DATA_TYPE>   & train_y,
		std::vector< std::vector <DATA_TYPE> >  & val_x,
		std::vector <DATA_TYPE>   & val_y,
		 int epochs, DATA_TYPE yida, DATA_TYPE reg);
	void training( 
		// cl::CommandQueue & cmd, 
		GPU & gpu,
		cl::Program &prog,
		std::vector< std::vector <DATA_TYPE> >& train_x,
		std::vector <DATA_TYPE> & train_y , 
		std::vector< std::vector <DATA_TYPE> >& val_x,
		std::vector <DATA_TYPE> & val_y , 
		int epochs, 
		int mini_batch,
		DATA_TYPE yida, 
		DATA_TYPE reg);

	DATA_TYPE predict(std::vector< std::vector <DATA_TYPE> >  & train_x,
		std::vector <DATA_TYPE>   & train_y) ;

// private:
	int N;   // # of data points per class
	int dim; // dimensionality
	int k;   // # of class
	int h1;  // size of hidden layer 1
	int h2;  // size of hidden layer 2
	std::vector< std::vector <DATA_TYPE> >  w1;
	std::vector< std::vector <DATA_TYPE> >  w2;
	std::vector< std::vector <DATA_TYPE> >  w3;

	std::vector <DATA_TYPE>  b1;
	std::vector <DATA_TYPE>  b2;
	std::vector <DATA_TYPE>  b3;

};

std::vector< std::vector< std::vector <DATA_TYPE> > > split( 
	std::vector< std::vector <DATA_TYPE> >& my_array, int dim, DATA_TYPE theta);


/*
	OpenCL operations
*/

void matrix_matrix( GPU &gpu, cl::Program &prog, const int  M,
	const int  K, const int  N, cl::Buffer &my_array_x_, cl::Buffer &w1,
	cl::Buffer &hidden_layer_1 );

void m_v_add( GPU &gpu, cl::Program &prog, const int  mini_batch,
	const int  K, cl::Buffer &hidden_layer_1, cl::Buffer &b1 );

void sigmoid( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &hidden_layer_1);

void softmax( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &scores);

void multinominal_cross_entropy(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int class_num,
	cl::Buffer &scores,
	cl::Buffer &temp_y,
	cl::Buffer &data_loss);

void gradient_score(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int class_num,
	cl::Buffer &scores,
	cl::Buffer &temp_y);

void transpose( 
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch, 
	const int h2,
	cl::Buffer & hidden_layer_2 , 
	cl::Buffer & hidden_layer_2_T );

void bias_sum( 
	GPU &gpu,
	cl::Program &prog, 
	const int class_num, 
	const int mini_batch,
	cl::Buffer & scores, 
	cl::Buffer & bias );

void sigmoid_grad( 
	GPU &gpu,
	cl::Program &prog, 
	const int  mini_batch,
	const int  K,
	cl::Buffer &hidden_layer_1);

void ele_multiply(
	GPU &gpu,
	cl::Program &prog, 
	const int mini_batch,
	const int K,
	cl::Buffer & array1, 
	cl::Buffer & array2 );

void update1(
	GPU &gpu,
	cl::Program &prog, 
	const int array1_x,
	const float yida,
	cl::Buffer & array1, 
	cl::Buffer & array2 );

void update2(
	GPU &gpu,
	cl::Program &prog, 
	const int array1_x,
	const int array1_y,
	const float yida,
	cl::Buffer & array1, 
	cl::Buffer & array2 );


/*
	Normal operations
*/
void sigmoid (std::vector< std::vector <DATA_TYPE> >& array);
void sigmoid_grad(std::vector< std::vector <DATA_TYPE> >& array);

std::vector< std::vector <DATA_TYPE> > matrix_matrix(std::vector< std::vector <DATA_TYPE> >& array1, 
	std::vector< std::vector <DATA_TYPE> >& array2 , bool transpose_1, bool transpose_2);

DATA_TYPE sum(std::vector< std::vector <DATA_TYPE> >& array1);

void update_1 ( std::vector <DATA_TYPE> & array1,
	std::vector <DATA_TYPE> & array2,
	DATA_TYPE b );

void update_2 (std::vector< std::vector <DATA_TYPE> >& array1,
	std::vector< std::vector <DATA_TYPE> >& array2,
	DATA_TYPE b );

void m_v_add (std::vector< std::vector <DATA_TYPE> >& array, std::vector <DATA_TYPE> & b );

void read_file( char *file,
	std::vector< std::vector <DATA_TYPE> >& my_array, 
	int max_fea);

// void read_file( char *file, DATA_TYPE * my_array , int max_fea);


void cv_split( 
	std::vector< std::vector <DATA_TYPE> > &my_array ,
	std::vector< std::vector <DATA_TYPE> > &my_array_x,
	std::vector <DATA_TYPE>  &my_array_y, 
	std::vector< std::vector <DATA_TYPE> > &val_x ,
	std::vector <DATA_TYPE>  &val_y, DATA_TYPE percent);


#endif