TARGET:=libqcu.so qcu_serial.o
LIB:=libqcu.so
INCLUDE:=$(shell pwd)/../../pyquda/include/qcu
OBJECTS:=qcu_serial.o
SRC:=qcu_serial.cpp
FLAGS:= -Ofast # -g
INSTALL_DIR:=$(shell pwd)/../../lib



${LIB}: ${OBJECTS}
	g++ -shared -o ${LIB} ${OBJECTS}

${OBJECTS}: ${SRC} # qcu.h
	g++ -c ${SRC}  ${FLAGS} -I ${INCLUDE} -fPIC

.PHONY: install clean
install:
	cp ${TARGET} ${INSTALL_DIR}/
clean:
	-rm ${TARGET}
