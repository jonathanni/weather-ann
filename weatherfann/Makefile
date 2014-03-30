CC=gcc

CFLAGS=-c -O3 -Wall -Wextra
LDFLAGS=-L/usr/local/lib -lnetcdf -lfann -lm
IFLAGS=

SOURCES=engine.c model.c logger.c error.c fannarray.c

OBJECTS=$(SOURCES:.c=.o)

EXECUTABLE=model

.SUFFIXES: .c

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

.c.o: $(SOURCES)
	$(CC) $(IFLAGS) $(CFLAGS) $< -o $@

clean:
	rm $(EXECUTABLE) *.o
