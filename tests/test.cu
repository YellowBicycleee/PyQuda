#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define QCU_ARCH_WMMA_ENABLED
#define QCU_ARCH_WMMA_SM80_ENABLED
#endif


#include <iostream>

using namespace std;

void test_macro() {
#ifdef QCU_ARCH_WMMA_SM80_ENABLED 
    cout << "enabled" << endl;
#else
    cout << "disabled" << endl;
#endif
}

int main() {
    test_macro();
    return 0;
}
