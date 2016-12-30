seamless_clone: seamless_cloning.o clone.o poisson_solver.o
	g++ -I /usr/local/include/eigen3 $$(pkg-config --cflags --libs opencv) seamless_cloning.o clone.o poisson_solver.o -O3 -o seamless_clone

seamless_cloning.o: seamless_cloning.cpp clone.h
	g++ -I /usr/local/include/eigen3 $$(pkg-config --cflags --libs opencv) -c seamless_cloning.cpp

clone.o: clone.cpp clone.h poisson_solver.h
	g++ -I /usr/local/include/eigen3 $$(pkg-config --cflags --libs opencv) -c clone.cpp

poisson_solver.o: poisson_solver.cpp poisson_solver.h
	g++ -I /usr/local/include/eigen3 $$(pkg-config --cflags --libs opencv) -c poisson_solver.cpp

clean:
	rm *.o
