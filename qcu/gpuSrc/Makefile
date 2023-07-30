TARGET:=libqcu.so
LIB:=libqcu.so
INCLUDE:=$(pwd)/../../pyquda/include
QCU_INCLUDE:=${INCLUDE}/qcu
SRC:=my_dslash_new.cu
FLAGS:=-arch=sm_70    # -g -G
INSTALL_DIR:=$(pwd)/../../lib



${LIB}: ${SRC}
    nvcc -shared -Xcompiler -fPIC -o ${LIB} ${SRC} -I${INCLUDE} ${FLAGS}

debug: ${SRC} test.cu
    nvcc -o debug ${SRC} test.cu -I${INCLUDE} ${FLAGS} -G -g

.PHONY: install clean

install:
	cp ${TARGET} ${INSTALL_DIR}/
clean:
	-rm ${TARGET}
