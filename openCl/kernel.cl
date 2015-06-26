

// References: http://ircc.fiu.edu/sc13/OpenCL01_Slides.pdf
// http://www.cedricnugteren.nl/tutorial.php?page=7
// http://stackoverflow.com/questions/12426061/how-to-pass-and-access-c-vectors-to-opencl-kernel


__kernel 
void transpose(
	const int M, 
	const int N,
	const __global float* hidden_layer_2 , 
	__global float* hidden_layer_2_T ){
	
	const int globalRow = get_global_id(0);
	if( globalRow >= M ) return;

	for(int j=0; j < N; j++){
		hidden_layer_2_T [ globalRow + M * j ] = hidden_layer_2[ globalRow * N + j ];
	}


}


__kernel 
void matrix_matrix(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

	// Thread identifiers
	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)

	// Compute a single element (loop over K)
	float acc = 0.0f;
	for (int k=0; k<K; k++) {
		acc += A[k*M + globalRow] * B[globalCol*K + k];
	}

	// Store the result
	C[globalCol*M + globalRow] = acc;
}


__kernel 
void m_v_add( 
	const int M,
	const int N,
	__global float* array, 
	__global float* b  ){
	
	int j;
	const int globalRow = get_global_id(0);
	if( globalRow >= M ) return;

	for( j=0; j< N; j++ ){
		array[ globalRow*N + j ] += b[j];
	}

}

__kernel
void sigmoid( 
	const int mini_batch,
	const int K,
	__global float* hidden_layer_1){

	int j;
	const int globalRow = get_global_id(0);
	if( globalRow >= mini_batch ) return;

	for( j=0; j< K; j++ ){
		hidden_layer_1[ globalRow*K + j ] = 1/( 1+ exp( -1.*hidden_layer_1[ globalRow*K + j ] ) );
	}

}

__kernel 
void softmax(
	const int mini_batch,
	const int K,
	__global float* scores){
	
	int j;
	const int globalRow = get_global_id(0);
	if( globalRow >= mini_batch ) return;

	float temp = 0.0f;
	for (int j=0; j<K; j++) {
		temp += exp( scores[ globalRow*K + j ] );
	}

	for (int j=0; j<K; j++) {
		scores[ globalRow*K + j ] = exp( scores[ globalRow*K + j ] );
	}

}

__kernel 
void multinominal_cross_entropy(
	const int mini_batch,
	const int class_num,
	const __global float* scores,
	const __global float* temp_y,
	__global float* data_loss
	){
	

	//const int globalRow = get_global_id(0);
	//if( globalRow >= mini_batch ) return;
	//scores[ globalRow  ];
	//data_loss[0] += log( scores[ globalRow * class_num  + temp_y[ globalRow ] ] );

	for (int i=0; i < mini_batch; ++i ){
		//data_loss[0] += log(scores [ i ] [ temp_y[i] ] ) ;
		data_loss[0] += log( scores [ i * class_num  + (int)temp_y[i] ] ) ;
	}
	
	//barrier(CLK_LOCAL_MEM_FENCE);

	data_loss[0] = (-1.)*data_loss[0] / mini_batch;

}

__kernel 
void gradient_score(
	const int mini_batch,
	const int class_num,
	__global float* scores,
	const __global float* temp_y
	){

	const int globalRow = get_global_id(0);
	if( globalRow >= mini_batch ) return;
	//scores[ globalRow  ];
	scores[ globalRow * class_num  + (int)temp_y[ globalRow ] ] -= 1;

	for (int i=0; i < class_num; ++i ){
		//data_loss[0] += log(scores [ i ] [ temp_y[i] ] ) ;
		scores[ globalRow * class_num  + i ] /= mini_batch;
	}

}

__kernel 
void bias_sum(
	const int class_num,
	const int mini_batch,
	const __global float* scores,
	__global float* bias){
	
	const int globalRow = get_global_id(0);
	if( globalRow >= class_num ) return;

	float temp = 0.0;
	for(int q=0; q< mini_batch ; q++){
		temp += scores[  q * class_num + globalRow ];
	}
	bias[ globalRow ] = temp;

}

__kernel
void sigmoid_grad( 
	const int mini_batch,
	const int K,
	__global float* hidden_layer_1){

	int j;
	const int globalRow = get_global_id(0);
	if( globalRow >= mini_batch ) return;

	for( j=0; j< K; j++ ){
		hidden_layer_1[ globalRow*K + j ] =  hidden_layer_1[ globalRow*K + j ] *(1-hidden_layer_1[ globalRow*K + j ]) ;
	}

}


__kernel 
void ele_multiply(
	const int mini_batch,
	const int K,
	__global float* dhidden2, 
	const __global float* hidden_layer_2 ){
	
	int j;
	const int globalRow = get_global_id(0);
	if( globalRow >= mini_batch ) return;

	for( j=0; j< K; j++ ){
		dhidden2[ globalRow*K + j ] *=  hidden_layer_2[ globalRow*K + j ] ;
	}

}

__kernel 
void update_1 (
	const int M,
	const float yida,
	__global float* array1, 
	const __global float* array2 ){
	
	const int globalRow = get_global_id(0);
	if( globalRow >= M ) return;

	array1[globalRow] += -1.*yida * array2[globalRow];

}

__kernel 
void update_2(
	const int M,
	const int N,
	const float yida,
	__global float* array1, 
	const __global float* array2 ){

	const int globalRow = get_global_id(0);
	if( globalRow >= M ) return;

	for(int k=0; k<N; k++){
		array1[ globalRow*N + k ] += -1.*yida * array2[globalRow*N + k];
	}

}

