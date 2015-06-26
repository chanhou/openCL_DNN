#include "YoUtil.hpp"
#include <iostream>			// for I/O
#include <fstream>			// for file I/O

using namespace std;

cl::Context *getContext(cl_device_type type, vector<cl::CommandQueue> &cmdQueues, cl_command_queue_properties props, bool verbose) {
	vector< cl::Platform > platforms;
	cl::Platform::get(&platforms);

	for (cl::Platform &platform : platforms) {
		vector<cl::Device> devices;
		platform.getDevices(type, &devices);
		if (devices.size() == 0) continue;

		cl::Context *context = new cl::Context(devices);

		// create command queues in the context
		cmdQueues.clear();
		for (auto& device : context->getInfo<CL_CONTEXT_DEVICES>()) {
			// cmdQueues.push_back(cl::CommandQueue(*context, device, props));
			cmdQueues.push_back(cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE));
		}

		if (verbose) {
			cout << "\n\tThere are " << context->getInfo<CL_CONTEXT_NUM_DEVICES>() << " device(s) in the defined context.";
			cout << "\n\t# of Command queues: " << cmdQueues.size();
		}
		return context;
	}

	if (verbose) cerr << "\nCannot find any platform for given device type: " << type;
	return nullptr;
}

string readSourceCode(const char *filename) {
	ifstream inp(filename);
	if (!inp) {
		cerr << "\nError opening file: " << filename << endl;
		return "";
	}

	string kernel((istreambuf_iterator<char>(inp)), istreambuf_iterator<char>());

	inp.close();
	return kernel;
}

cl::Program *compile(cl::Context &context, const string &source, const char *options) {
	cl::Program *prog = new cl::Program(context, source);

	try {
		// prog->build("-cl-std=CL1.2 -w -cl-kernel-arg-info");
		prog->build(options);
	}
	catch (cl::Error &e) {
		cerr << "\nFile: " << __FILE__ << ", line: " << __LINE__ << e.what();
		cerr << "\nError no: " << e.err() << endl;
		for (auto& device : context.getInfo<CL_CONTEXT_DEVICES>()) {
			cout << "\n=== " << device.getInfo<CL_DEVICE_NAME>() << " ===";
			cout << "\nBuild log: " << prog->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			cout << "\nBuild options used:" << prog->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
		}
		return nullptr;
	}

	/*
	// See Table 5.13 of OpenCL 1.2 specification for information that can be queried to program objects
	cout << "\n\t# devices associated with the program: " << prog->getInfo<CL_PROGRAM_NUM_DEVICES>();
	cout << "\n\t# Kernels defined: " << prog->getInfo<CL_PROGRAM_NUM_KERNELS>();
	cout << "\n\tProgram kernel names: " << prog->getInfo<CL_PROGRAM_KERNEL_NAMES>();
	cout << "\n\tProg sizes: ";  for (auto s : prog->getInfo<CL_PROGRAM_BINARY_SIZES>()) cout << s << ";";
	*/

	return prog;
}
