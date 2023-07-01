objects = ../so/Strategy.so ../so/Strategy.so.d

so:		# Make so for local test
	g++ -Wall -std=c++11 -O2 -fpic -shared Judge.cpp Strategy.cpp -o ../so/Strategy.so

debug:	# Make so with -DDEBUG and -O0 for debug
	# **Notice that output result is Strategy.so.d**
	g++ -Wall -std=c++11 -O0 -DDEBUG -fpic -shared Judge.cpp Strategy.cpp -o ../so/Strategy.so.d

clean:
	rm -f $(objects)
