#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include "dnn.h"

using namespace std;

int main(int argc, char** argv){

	vector< vector <double> > my_array;
	vector< vector <double> > my_array_x;
	vector <double>  my_array_y;
	vector< vector <double> > val_x;
	vector <double>  val_y;
	// vector< vector< vector <double> > > eee;
	// vector<int> label_array;

	vector< vector <double> > testing;
	vector< vector <double> > test_x;
	vector <double>  test_y;
	vector< vector <double> > test_val_x;
	vector <double>  test_val_y;


	srand (time(NULL));	

	read_file( argv[8], my_array, 123);
	cv_split( my_array, my_array_x, my_array_y , val_x , val_y, 0.0);

	read_file( argv[9], testing, 123);
	cv_split( testing, test_x, test_y , test_val_x , test_val_y, 0.0);
	// cout<<my_array_x.size()<<" "<<my_array_x[0].size()<<endl;
	// dnn * net = new dnn( my_array_x.size() , my_array_x[0].size() , 2, 100,100);

	// for(auto &x: my_array_y){
	// 		cout<< x<<" ";
	// }

	//              dnn( data_num, dimension, class, hidd1, hidd2 );
	dnn * net = new dnn( my_array_x.size() , atoi(argv[1]) , atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
	//                                                    epochs 		 learning rate  weight norm
	net->training(my_array_x, my_array_y, test_x, test_y, atoi(argv[5]), atof(argv[6]), atof(argv[7]));


	return 0;
}