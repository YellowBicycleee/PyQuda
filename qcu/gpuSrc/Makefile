TARGET:=libqcu.so
LIB:=libqcu.so
INCLUDE:=$(shell pwd)/../../pyquda/include/qcu
SRC:=my_dslash_new.cu dslash_eo.cu
FLAGS:=-arch=sm_70    # -g -G
INSTALL_DIR:=$(shell pwd)/../../lib



${LIB}: ${SRC}
	nvcc -shared -Xcompiler -fPIC -o ${LIB} ${SRC} -I${INCLUDE} ${FLAGS}

test: ${SRC} test.cu
	nvcc -o test ${SRC} test.cu -I${INCLUDE} ${FLAGS} -G -g

.PHONY: install clean

install:
	cp ${TARGET} ${INSTALL_DIR}/
clean:
	-rm ${TARGET}
