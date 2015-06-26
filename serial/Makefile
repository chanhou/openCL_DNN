target = dnn

CPPFLAGS = -O2 -std=c++11 
# -Wall
all: $(target)

%: %.cpp
	g++ $(CPPFLAGS) $< -o $@

dnn: dnn_main.cpp dnn.cpp dnn.h stopWatch.o
	g++ $(CPPFLAGS) $^ -o $@

test: test.cpp
	g++ $(CPPFLAGS) $^ -o $@

run_dnn:
	# ./dnn ./homework4-forests/heart.train ./homework4-forests/heart.test
	./dnn 123 2 500 500 1 0.01 0.01 ./homework4-forests/a1a.test ./homework4-forests/a1a.train
	# ./dnn 123 2 20 20 500 0.01 0.01 ./homework4-forests/a2a.train ./homework4-forests/a2a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a3a.train ./homework4-forests/a3a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a4a.train ./homework4-forests/a4a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a5a.train ./homework4-forests/a5a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a6a.train ./homework4-forests/a6a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a7a.train ./homework4-forests/a7a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a8a.train ./homework4-forests/a8a.t
	# ./dnn 123 2 10 10 20 0.01 0.01 ./homework4-forests/a9a.train ./homework4-forests/a9a.t
	

# run_test:
# 	./test ./homework4-forests/heart.train
