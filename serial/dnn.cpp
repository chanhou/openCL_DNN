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

#include "stopWatch.h"
#include "dnn.h"

#include <cassert>
#include <chrono>

using namespace std;

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

	std::normal_distribution<double> distribution (0.0,1.0);
	

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


void read_file(char *file, vector< vector <double> >& my_array , 
	int max_fea){

	double* features = new double[MAX_FEATURE];
	
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
		memset(features, 0, sizeof(double) * MAX_FEATURE);  // replace the value of the memory size

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

	// double* my_array = new double[max_index*num_data];
	// int* label_array = new int[num_data];

	// vector< vector <double> > my_array(num_data, vector<double>(max_index,0));
	// vector< vector <double> > my_array;
	// vector<int> label_array(num_data,0);
	// vector<int> label_array;

	// my_array(num_data, vector<double>(max_index,0));
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
		// memset(my_array, 0, sizeof(double) * MAX_FEATURE);  // replace the value of the memory size

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
	std::vector< std::vector <double> > &my_array ,
	std::vector< std::vector <double> > &my_array_x,
	std::vector <double>  &my_array_y, 
	std::vector< std::vector <double> > &val_x ,
	std::vector <double>  &val_y, double percent){

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

	for(int i=0; i< my_array.size(); ++i){
		if (i<num_train)
			my_array_y[i] = my_array[i][0];
		else
			val_y[i - num_train] = my_array[i][0];

		for (int q = 0; q< max_index; ++q){
			if(i< num_train)
				my_array_x[i][q] = my_array[i][q+1];
			else
				val_x[i - num_train ][q] = my_array[i][q+1];
		}
	}

}

vector< vector< vector <double> > > split( vector< vector <double> >& my_array, int dim, double theta){
	
	vector< vector< vector <double> > > sppp(2);

	for(int i=0 ; i<(int) my_array.size(); ++i){
		if(my_array[i][dim] < theta ){
			sppp[0].push_back( my_array[i] );
		}
		else{
			sppp[1].push_back( my_array[i] );
		}
	}

	return sppp;
}

void dnn::training( vector< vector <double> >& train_x,
	vector <double> & train_y , 
	vector< vector <double> >& val_x,
	vector <double> & val_y , 
	int epochs, double yida, double reg){

	vector < vector <double>>  hidden_layer_1;
	vector < vector <double>>  hidden_layer_2;
	vector < vector <double>>  scores;

	vector < vector <double>>  temp;

	vector < vector <double>>  dw3;
	vector < vector <double>>  dw2;	
	vector < vector <double>>  dw1;
	vector <double>  db3;
	vector <double>  db2;
	vector <double>  db1;

	vector < vector <double>>  dhidden2;
	vector < vector <double>>  dhidden1;

	double num_examples = (double)train_x.size();

	stopWatch timer;

	for(int iter=0; iter< epochs; ++iter){
		double t1;
		
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
		for (int i=0 ; i<scores.size();++i){
			double temp = 0;
			for(int j=0; j< scores[0].size(); ++j){ // class num
				temp += exp(scores[i][j]); // normalize term
			}
			for (int j=0; j< scores[0].size(); ++j){
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

		double data_loss = 0;
		for(int i=0; i<scores.size();++i){
			data_loss += log(scores[i][train_y[i]]) ;
		}
		data_loss = (-1)*data_loss / num_examples;
		
		double reg_loss = 0;

		temp = matrix_matrix(w1,w1,true,false);
		reg_loss += 0.5*reg*sum(temp);

		temp = matrix_matrix(w2,w2,true,false);
		reg_loss += 0.5*reg*sum(temp);

		temp = matrix_matrix(w3,w3,true,false);
		reg_loss += 0.5*reg*sum(temp);

		data_loss = data_loss + reg_loss;

		// # compute the gradient on scores
		for(int i=0;i<scores.size(); ++i){
			scores[i][ train_y[i] ] -= 1;
			for(int q=0;q<scores[0].size(); ++q)
				scores[i][q] /= num_examples;
		}

		// # BACKPROP HERE
		// dw3 = (hidden_layer_2.T).dot(dscores)
		// db3 = np.sum(dscores, axis=0, keepdims=True)

		dw3 = matrix_matrix(hidden_layer_2, scores, true,false);

		for(int i=0; i<scores[0].size(); ++i){
			db3.push_back(0);
			for(int q=0; q< scores.size(); ++q){
				db3[i] += scores[q][i];
			}
		}

		// #backprop sigmoid nonlinearity here
		// dhidden2 = dscores.dot(w3.T)*sigmoid_grad(hidden_layer_2)

		dhidden2 = matrix_matrix( scores, w3, false, true );

		sigmoid_grad(hidden_layer_2);

		for(int i=0; i< dhidden2.size(); ++i){
			for(int q =0; q< dhidden2[0].size(); ++q){
				dhidden2[i][q] *= hidden_layer_2[i][q];
			}
		}

		// dw2 = (hidden_layer.T).dot(dhidden2)
		// db2 = np.sum(dhidden2, axis=0)
		dw2 = matrix_matrix(hidden_layer_1 , dhidden2, true , false);
		for(int i=0; i<dhidden2[0].size(); ++i){
			db2.push_back(0);
			for(int q=0; q< dhidden2.size(); ++q){
				db2[i] += dhidden2[q][i];
			}
		}

		// dhidden = dhidden2.dot(w2.T)*sigmoid_grad(hidden_layer_1)
		dhidden1 = matrix_matrix(dhidden2, w2, false, true);
		sigmoid_grad(hidden_layer_1);
		for(int i=0; i< dhidden1.size(); ++i){
			for(int q =0; q< dhidden1[0].size(); ++q){
				dhidden1[i][q] *= hidden_layer_1[i][q];
			}
		}

		// dw1 =  np.dot(X.T, dhidden)
		// db1 = np.sum(dhidden, axis=0)

		// cout<<train_x.size()<<" "<<train_x[0].size()<<endl;
		// cout<<dhidden1.size()<<" "<<dhidden1[0].size()<<endl;
		
		dw1 = matrix_matrix(train_x, dhidden1, true, false);

		for(int i=0; i<dhidden1[0].size(); ++i){
			db1.push_back(0);
			for(int q=0; q< dhidden1.size(); ++q){
				db1[i] += dhidden1[q][i];
			}
		}

		// # add regularization
		// dW3+= reg * W3
		// dW2 += reg * W2
		// dW1 += reg * W1
		for(int i=0; i<dw3.size(); ++i)
			for (int q=0; q<dw3[0].size(); ++q)
				dw3[i][q] += reg * w3[i][q];
		for(int i=0; i<dw2.size(); ++i)
			for (int q=0; q<dw2[0].size(); ++q)
				dw2[i][q] += reg * w2[i][q];
		for(int i=0; i<dw1.size(); ++i)
			for (int q=0; q<dw1[0].size(); ++q)
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
		cout<<t1<<endl;
		// if (iter%10==0){
		// 	printf("epochs %d: loss %f, val: %f, t: %f \n", 
		// 		iter, data_loss , predict(val_x, val_y), t1 );
		// }	

	}




}

double dnn::predict( std::vector< std::vector <double> >  & train_x,
		std::vector <double>   & train_y ){
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

	vector < vector <double>>  hidden_layer_1;
	vector < vector <double>>  hidden_layer_2;
	vector < vector <double>>  scores;

	vector <int>  predict_class;
	
	hidden_layer_1 = ( matrix_matrix(train_x, w1, false, false) );

	m_v_add(hidden_layer_1, b1);
	sigmoid(hidden_layer_1);

	hidden_layer_2 = matrix_matrix(hidden_layer_1, w2, false, false);
	m_v_add(hidden_layer_2, b2);
	sigmoid(hidden_layer_2);

	scores = matrix_matrix(hidden_layer_2, w3, false, false);
	m_v_add(scores, b3);
	
	for(int i=0; i< scores.size(); ++i){
		predict_class.push_back(0);
		double temp = -INFINITY;
		for (int q=0; q<scores[0].size(); ++q){
			if (scores[i][q]>temp){
				temp = scores[i][q];
				predict_class[i] = q; // y label
			}
		}
	}

	double correct =0;
	for (int i=0; i<predict_class.size(); ++i){
		if (predict_class[i] == train_y[i])
			correct++;
	}

	return correct/(double)predict_class.size();

}

void sigmoid (std::vector< std::vector <double> >& array){
	// x = 1/(1+np.exp(-x))
	for (int x=0; x < array.size(); ++x){
		for(int y = 0; y<array[0].size(); ++y){
			array[x][y] = 1 / ( 1 + exp(-1*array[x][y]) );
		}
	}
}

void sigmoid_grad(std::vector< std::vector <double> >& array){
	// (x)*(1-x)
	for (int x=0; x < array.size(); ++x){
		for(int y = 0; y<array[0].size(); ++y){
			array[x][y] = array[x][y] * ( 1- array[x][y] );
		}
	}
}

double sum(std::vector< std::vector <double> >& array1){
	double sum=0;
	for (int i=0;i<array1.size(); ++i){
		for(int q=0; q< array1[0].size(); ++q){
			sum += array1[i][q];
		}
	}
	return sum;
}

void update_1 ( std::vector <double> & array1,
	std::vector <double> & array2,
	double b ){

	for (int i=0; i< array1.size(); ++i){
			array1[i] += -b* array2[i] ;
	}
}

void update_2 (std::vector< std::vector <double> >& array1,
	std::vector< std::vector <double> >& array2,
	double b ){
	for (int i=0; i< array1.size(); ++i){
		for(int q =0 ;q < array1[0].size(); ++q){
			array1[i][q] += -b * array2[i][q] ;
		}
	}
}



void m_v_add (std::vector< std::vector <double> >& array,
	std::vector <double> & b ){

	assert(array[0].size()==b.size());
	// cout<<"==="<<endl;
	for (int i=0; i< array.size(); ++i){ // 20
		for(int q =0 ;q < array[0].size(); ++q){ //100
			array[i][q] += b[q];
		}
	}
}

std::vector< std::vector <double> > matrix_matrix(std::vector< std::vector <double> >& array1, 
	std::vector< std::vector <double> >& array2, bool transpose_1, bool transpose_2){

	int x = array1.size();
	int y = array2[0].size();

	if (transpose_1){
		
		int z = array1[0].size();
		vector< vector <double> > array1_T(z, vector<double> (x,0));
		for(int i=0; i<x; ++i ){
			for(int j=0; j<z; ++j){
				array1_T[j][i] = array1[i][j];
			}
		}

		// array1 transpose
		assert(array1.size()==array2.size());

		// cout<<x<<" "<<array1[0].size()<<" "<<array2.size()<<" "<<y<<endl;
		vector< vector <double> > result(z, vector<double> (y,0));
		for (int i=0; i < array1_T.size(); ++i){
			for(int k = 0; k<array2[0].size(); ++k){
				for (int j=0; j< array1_T[0].size(); ++j){
					result[i][k] += array1_T[i][j] * array2[j][k];
				}
			}
		}

		array1_T.clear();
		return result;
	}
	else if (transpose_2){
		int z = array2.size();
		vector< vector <double> > array2_T(y, vector<double> (z,0));
		for(int i=0; i<z; ++i ){
			for(int j=0; j<y; ++j){
				array2_T[j][i] = array2[i][j];
			}
		}

		// array 2 transpose
		assert(array1[0].size()==array2[0].size());
		// cout<<x<<" "<<array1[0].size()<<" "<<array2.size()<<" "<<y<<endl;
		vector< vector <double> > result(x, vector<double> (z,0));
		for (int i=0; i < array1.size(); ++i){
			for(int k = 0; k<array2_T[0].size(); ++k){
				for (int j=0; j< array1[0].size(); ++j){
					result[i][k] += array1[i][j] * array2_T[j][k];
				}
			}
		}

		array2_T.clear();
		return result;
	}
	else{

		assert(array1[0].size()==array2.size());

		vector< vector <double> > result(x, vector<double> (y,0));
		for (int i=0; i < array1.size(); ++i){
			for(int k = 0; k<array2[0].size(); ++k){
				for (int j=0; j< array1[0].size(); ++j){
					result[i][k] += array1[i][j] * array2[j][k];
				}
			}
		}
		return result;
	}

	
	// cout<<result.size()<<" "<<result[0].size()<<endl;
	
}

