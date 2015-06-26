#ifndef __STOPWATCH
#define __STOPWATCH

#ifdef WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
	#include <stddef.h>
#endif

class stopWatch {
private:
#ifdef WIN32
	LARGE_INTEGER t1, t2, tick;
#else
	struct timeval t1, t2;
#endif	
public:
	stopWatch();
	void start(void);
	void stop(void);
	double elapsedTime(void);
};

#endif
