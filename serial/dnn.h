#ifndef DNN
#define DNN

#include <vector>
#include <cstddef>
#include <stdio.h>
#include <stack>

/*
reference:
1. http://cs224d.stanford.edu/notebooks/vanishing_grad_example.html
*/

class dnn {
public:
	dnn( int num, int dimension, int klass, int hidd1, int hidd2 );
	~dnn();

	void print_file( FILE * abc ) const;

	void training(
		std::vector< std::vector <double> >  & train_x,
		std::vector <double>   & train_y,
		std::vector< std::vector <double> >  & val_x,
		std::vector <double>   & val_y,
		 int epochs, double yida, double reg);

	double predict(std::vector< std::vector <double> >  & train_x,
		std::vector <double>   & train_y) ;

private:
	int N;   // # of data points per class
	int dim; // dimensionality
	int k;   // # of class
	int h1;  // size of hidden layer 1
	int h2;  // size of hidden layer 2
	std::vector< std::vector <double> >  w1;
	std::vector< std::vector <double> >  w2;
	std::vector< std::vector <double> >  w3;

	std::vector <double>  b1;
	std::vector <double>  b2;
	std::vector <double>  b3;

};

std::vector< std::vector< std::vector <double> > > split( 
	std::vector< std::vector <double> >& my_array, int dim, double theta);

void sigmoid (std::vector< std::vector <double> >& array);
void sigmoid_grad(std::vector< std::vector <double> >& array);

std::vector< std::vector <double> > matrix_matrix(std::vector< std::vector <double> >& array1, 
	std::vector< std::vector <double> >& array2 , bool transpose_1, bool transpose_2);

double sum(std::vector< std::vector <double> >& array1);

void update_1 ( std::vector <double> & array1,
	std::vector <double> & array2,
	double b );

void update_2 (std::vector< std::vector <double> >& array1,
	std::vector< std::vector <double> >& array2,
	double b );

void m_v_add (std::vector< std::vector <double> >& array, std::vector <double> & b );

void read_file( char *file,std::vector< std::vector <double> >& my_array, int max_fea);

void cv_split( 
	std::vector< std::vector <double> > &my_array ,
	std::vector< std::vector <double> > &my_array_x,
	std::vector <double>  &my_array_y, 
	std::vector< std::vector <double> > &val_x ,
	std::vector <double>  &val_y, double percent);


#endif