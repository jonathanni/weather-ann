CC=gcc
NVCC=/usr/local/cuda/bin/nvcc

# NVCCFLAGS must have -O2 or IT WILL GENERATE BUGS!!!
CFLAGS=-c -O3 -Wall -Wextra
NVCCFLAGS=-c -O2 -Xcompiler -fno-aggressive-loop-optimizations -gencode=arch=compute_20,code=sm_20
LDFLAGS=-L/usr/local/lib -L/usr/local/cuda/lib64 -lnetcdf -lm -lcuda -lcudart
IFLAGS=-I/usr/local/cuda/include

SOURCES=model.c logger.c error.c
CUSOURCES=engine.cu

OBJECTS=$(SOURCES:.c=.o) $(CUSOURCES:.cu=.o)

EXECUTABLE=model

.SUFFIXES: .c .cu

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

.c.o: $(SOURCES)
	$(CC) $(IFLAGS) $(CFLAGS) $< -o $@

.cu.o: $(CUSOURCES)
	$(NVCC) $(IFLAGS) $(NVCCFLAGS) $< -o $@

clean:
	rm $(EXECUTABLE) *.o
