target = dnn

CPPFLAGS = -Wall -O3 -std=c++11 -march=native -funroll-loops -fomit-frame-pointer -fopenmp
# CPPFLAGS = -O2 -std=c++11 

iOCL=-I. -I/opt/AMDAPP/current/include -I/opt/AMDAPP/current/include/CL
lOCL=-L/opt/AMDAPP/current/lib/x86_64 -lOpenCL

# -Wall
all: $(target)

# %: %.cpp
# 	g++ $(CPPFLAGS) $< -o $@

YoUtil.o: YoUtil.cpp
	g++ $(CPPFLAGS) -c $< -o $@

# dnn: dnn_main.cpp dnn.cpp dnn.h stopWatch.o
# 	g++ $(CPPFLAGS) $^ -o $@

dnn: dnn_main.cpp dnn.cpp dnn.h YoUtil.o
	g++ $(CPPFLAGS) dnn_main.cpp dnn.cpp dnn.h -o $@ $(iOCL) $(lOCL) -lrt YoUtil.o

test: test.cpp
	g++ $(CPPFLAGS) $^ -o $@

run_dnn:
	# ./dnn ./homework4-forests/heart.train ./homework4-forests/heart.test
	./dnn 123 2 500 500 1 0.01 0.01 ./homework4-forests/a1a.test ./homework4-forests/a1a.train
	# ./dnn ./homework4-forests/a2a.train ./homework4-forests/a2a.t
	# ./dnn ./homework4-forests/a3a.train ./homework4-forests/a3a.t
	# ./dnn ./homework4-forests/a4a.train ./homework4-forests/a4a.t
	# ./dnn ./homework4-forests/a5a.train ./homework4-forests/a5a.t
	# ./dnn ./homework4-forests/a6a.train ./homework4-forests/a6a.t
	# ./dnn ./homework4-forests/a7a.train ./homework4-forests/a7a.t
	# ./dnn ./homework4-forests/a8a.train ./homework4-forests/a8a.t
	# ./dnn ./homework4-forests/a9a.train ./homework4-forests/a9a.t
	

# run_test:
# 	./test ./homework4-forests/heart.train