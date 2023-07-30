TARGET:=libqcu.so qcu_serial.o
LIB:=libqcu.so
INCLUDE:=$(pwd)/../../pyquda/include
QCU_INCLUDE:=${INCLUDE}/qcu
OBJECTS:=qcu_serial.o
SRC:=qcu_serial.cpp
FLAGS:= -Ofast # -g
INSTALL_DIR:=$(pwd)/../../lib



${LIB}: ${OBJECTS}
    g++ -shared -o ${LIB} ${OBJECTS} -fPIC

${OBJECTS}: ${SRC} # qcu.h
    g++ -c ${SRC}  ${FLAGS} -I ${QCU_INCLUDE}

.PHONY: install clean
install:
	cp ${TARGET} ${INSTALL_DIR}/
clean:
	-rm ${TARGET}
