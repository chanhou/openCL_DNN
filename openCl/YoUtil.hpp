#ifndef YO_OPENCL_UTIL_HEADER
#define YO_OPENCL_UTIL_HEADER

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <string>			// for C++ string
#include <vector>			// for vector container
#include <chrono>
using namespace std;

// utility function get a context by specifying device type
cl::Context *getContext(cl_device_type type, vector<cl::CommandQueue> &cmdQueues, cl_command_queue_properties props = 0, bool verbose = false);

// utility function to load OpenCL source code and store it into a string
std::string readSourceCode(const char *filename);

// utility function compile the CL source code into Programs
cl::Program *compile(cl::Context &, const string &source, const char *option = "-cl-std=CL1.2 -w -cl-kernel-arg-info");

// utility class for stopWatch wrapper using c++11 chrono
class stopWatch {
	chrono::high_resolution_clock::time_point t_start, t_stop;
public:
	void start() {
		t_start = chrono::high_resolution_clock::now();
	}
	void stop() {
		t_stop = chrono::high_resolution_clock::now();
	}
	double elapsedTime() {
		chrono::duration<double> d = t_stop - t_start;
		return d.count();
	}
	static double resolution() {
		auto tmp = chrono::high_resolution_clock::period();
		return (double)tmp.num / tmp.den;
	}
};

/*

Device Type:                                   CL_DEVICE_TYPE_GPU
Device ID:                                     4098
Board name:                                    AMD Radeon HD 7900 Series
Device Topology:                               PCI[ B#1, D#0, F#0 ]
Max compute units:                             32
Max work items dimensions:                     3
Max work items[0]:                           256
Max work items[1]:                           256
Max work items[2]:                           256
Max work group size:                           256

Preferred vector width char:                   4
Preferred vector width short:                  2
Preferred vector width int:                    1
Preferred vector width long:                   1
Preferred vector width float:                  1
Preferred vector width double:                 1
Native vector width char:                      4
Native vector width short:                     2
Native vector width int:                       1
Native vector width long:                      1
Native vector width float:                     1
Native vector width double:                    1
Max clock frequency:                           1000Mhz
Address bits:                                  32
Max memory allocation:                         1073741824
Image support:                                 Yes
Max number of images read arguments:           128
Max number of images write arguments:          8
Max image 2D width:                            16384
Max image 2D height:                           16384
Max image 3D width:                            2048
Max image 3D height:                           2048
Max image 3D depth:                            2048
Max samplers within kernel:                    16
Max size of kernel argument:                   1024
Alignment (bits) of base address:              2048
Minimum alignment (bytes) for any datatype:    128
Single precision floating point capability
Denorms:                                     No
Quiet NaNs:                                  Yes
Round to nearest even:                       Yes
Round to zero:                               Yes
Round to +ve and infinity:                   Yes
IEEE754-2008 fused multiply-add:             Yes
Cache type:                                    Read/Write
Cache line size:                               64
Cache size:                                    16384
Global memory size:                            3107979264
Constant buffer size:                          65536
Max number of constant args:                   8
Local memory type:                             Scratchpad
Local memory size:                             32768
Kernel Preferred work group size multiple:     64
Error correction support:                      0
Unified memory for Host and Device:            0
Profiling timer resolution:                    1
Device endianess:                              Little
Available:                                     Yes
Compiler available:                            Yes
Execution capabilities:
Execute OpenCL kernels:                      Yes
Execute native function:                     No
Queue properties:
Out-of-Order:                                No
Profiling :                                  Yes
Platform ID:                                   0x00007f3a8eed7500
Name:                                          Tahiti
Vendor:                                        Advanced Micro Devices, Inc.
Device OpenCL C version:                       OpenCL C 1.2
Driver version:                                1411.4 (VM)
Profile:                                       FULL_PROFILE
Version:                                       OpenCL 1.2 AMD-APP (1411.4)
Extensions:                                    cl_khr_fp64 cl_amd_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_gl_sharing cl_ext_atomic_counters_32 cl_amd_device_attribute_query cl_amd_vec3 cl_amd_printf cl_amd_media_ops cl_amd_media_ops2 cl_amd_popcnt cl_khr_image2d_from_buffer cl_khr_spir


*/


#endif