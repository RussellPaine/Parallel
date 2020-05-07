objects = main.o kernal.o

all: $(objects)
		nvcc $(objects) -o ./Builds/a.out -I=./Dependencies/include -lglut,GL -L./Dependencies/lib -l:libGLEW.a --compiler-options -no-pie

%.o: %.cu
		nvcc -x cu -I=./Dependencies/include  -dc $< -o $@

%.o: %.cpp
		nvcc -x cu -I=./Dependencies/include -dc $< -o $@

clean:
		rm -f *.o ./Builds/a.out
