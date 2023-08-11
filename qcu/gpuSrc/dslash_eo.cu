#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>

#include "qcu.h"

#define Nc 3
#define Nd 4
#define Ns 4
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define checkCudaErrors(err)                                                                                          \
  {                                                                                                                   \
    if (err != cudaSuccess) {                                                                                         \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                       \
    }                                                                                                                 \
  }

#define getVecAddr(origin, x, y, z, t, Lx, Ly, Lz, Lt)                                                                \
  ((origin) + ((((t) * (Lz) + (z)) * (Ly) + (y)) * (Lx) + (x)) * Ns * Nc) // 9 times
#define getGaugeAddr(origin, direction, x, y, z, t, Lx, Ly, Lz, Lt, parity)                                           \
  ((origin) + (direction) * (Lt) * (Lz) * (Ly) * (Lx)*2 * Nc * Nc + (parity) * ((Lt) * (Lz) * (Ly) * (Lx)*Nc * Nc) +  \
   ((((t) * (Lz) + (z)) * (Ly) + (y)) * (Lx) + (x)) * Nc * Nc)

class Complex {
private:
  double real_;
  double imag_;

public:
  __device__ __host__ Complex(double real, double imag) : real_(real), imag_(imag) {}
  Complex() = default;
  __device__ __host__ Complex(const Complex &complex) : real_(complex.real_), imag_(complex.imag_) {}
  __device__ __host__ double norm2() { return sqrt(real_ * real_ + imag_ * imag_); }
  __device__ __host__ void setImag(double imag) { imag_ = imag; }
  __device__ __host__ void setReal(double real) { real_ = real; }
  __device__ __host__ double real() const { return real_; }
  __device__ __host__ double imag() const { return imag_; }

  __device__ __host__ Complex &operator=(const Complex &complex)
  {
    real_ = complex.real_;
    imag_ = complex.imag_;
    return *this;
  }
  __device__ __host__ Complex &operator=(double rhs)
  {
    real_ = rhs;
    imag_ = 0;
    return *this;
  }
  __device__ __host__ Complex operator+(const Complex &complex) const
  {
    return Complex(real_ + complex.real_, imag_ + complex.imag_);
  }
  __device__ __host__ Complex operator-(const Complex &complex) const
  {
    return Complex(real_ - complex.real_, imag_ - complex.imag_);
  }
  __device__ __host__ Complex operator-() const { return Complex(-real_, -imag_); }
  __device__ __host__ Complex operator*(const Complex &rhs) const
  {
    return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
  }
  __device__ __host__ Complex &operator*=(const Complex &rhs)
  {
    real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
    imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
    return *this;
  }
  __device__ __host__ Complex operator/(const double &rhs) { return Complex(real_ / rhs, imag_ / rhs); }

  __device__ __host__ Complex &operator+=(const Complex &rhs)
  {
    real_ += rhs.real_;
    imag_ += rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &operator-=(const Complex &rhs)
  {
    real_ -= rhs.real_;
    imag_ -= rhs.imag_;
    return *this;
  }

  __device__ __host__ Complex &clear2Zero()
  {
    real_ = 0;
    imag_ = 0;
    return *this;
  }
  __device__ __host__ Complex conj() { return Complex(real_, -imag_); }
  __device__ __host__ bool operator==(const Complex &rhs) { return real_ == rhs.real_ && imag_ == rhs.imag_; }
  __device__ __host__ bool operator!=(const Complex &rhs) { return real_ != rhs.real_ || imag_ != rhs.imag_; }
  __device__ __host__ void output() const { printf("(%lf, %lf)", real_, imag_); }
};


__global__ void gpuDslash(void *U_ptr, void *a_ptr, void *b_ptr, int Lx, int Ly, int Lz, int Lt, int parity)
{
  assert(parity == 0 || parity == 1);

  __shared__ double shared_output_vec[BLOCK_SIZE * Ns * Nc * 2];
  double* shared_dest = static_cast<double*>(b_ptr) + (blockIdx.x * BLOCK_SIZE) * Ns * Nc * 2;
  // clear result shared memory
  for (int i = threadIdx.x; i < ((BLOCK_SIZE * Ns * Nc) << 1); i += BLOCK_SIZE) {
    shared_output_vec[i] = 0;
  }

  Complex *dest_temp;   // the beginning address of result vectors on (x,y,z,t)
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  Lx >>= 1;
  int t = thread / (Lx * Ly * Lz);
  thread -= t * (Lx * Ly * Lz);
  int z = thread / (Lx * Ly);
  thread -= z * (Lx * Ly);
  int y = thread / Lx;
  int x = thread - y * Lx;

  int eo = (t + z + y) & 0x1;  //   int eo = (t + z + y) % 2;
  int pos_x;
  Complex *u;
  Complex *res;
  Complex u_temp[Nc * Nc];    // for GPU
  Complex res_temp[Ns * Nc];  // for GPU

  Complex temp;

  __syncthreads();  // to sync because of clearing shared memory, delay to hide latency
  dest_temp = reinterpret_cast<Complex *>(shared_output_vec) + (threadIdx.x * Ns * Nc);

  // \mu = 1
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 0, x, y, z, t, Lx, Ly, Lz, Lt, parity);
  // #pragma unroll
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();
  pos_x = (1 & ~(parity ^ eo)) * x + (parity ^ eo) * ((x + 1) % Lx);
  res = getVecAddr(static_cast<Complex *>(a_ptr), pos_x, y, z, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] - res_temp[3 * Nc + j] * Complex(0, 1)) * u_temp[i * Nc + j];
      dest_temp[0 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] - res_temp[2 * Nc + j] * Complex(0, 1)) * u_temp[i * Nc + j];
      dest_temp[1 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp * Complex(0, 1);
    }
  }
  pos_x = (1 & ~(parity ^ eo)) * ((x - 1 + Lx) % Lx) + (parity ^ eo) * x;
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 0, pos_x, y, z, t, Lx, Ly, Lz, Lt, 1 - parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();

  res = getVecAddr(static_cast<Complex *>(a_ptr), pos_x, y, z, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] + res_temp[3 * Nc + j] * Complex(0, 1)) *
             u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[0 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] + res_temp[2 * Nc + j] * Complex(0, 1)) *
             u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[1 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp * Complex(0, -1);
    }
  }
  // \mu = 2
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 1, x, y, z, t, Lx, Ly, Lz, Lt, parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();
  res = getVecAddr(static_cast<Complex *>(a_ptr), x, (y + 1) % Ly, z, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] + res_temp[3 * Nc + j]) * u_temp[i * Nc + j];
      dest_temp[0 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp;
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] - res_temp[2 * Nc + j]) * u_temp[i * Nc + j];
      dest_temp[1 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += -temp;
    }
  }
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 1, x, (y + Ly - 1) % Ly, z, t, Lx, Ly, Lz, Lt, 1 - parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();

  res = getVecAddr(static_cast<Complex *>(a_ptr), x, (y + Ly - 1) % Ly, z, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] - res_temp[3 * Nc + j]) * u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[0 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += -temp;
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] + res_temp[2 * Nc + j]) * u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[1 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp;
    }
  }
  // \mu = 3
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 2, x, y, z, t, Lx, Ly, Lz, Lt, parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();
  res = getVecAddr(static_cast<Complex *>(a_ptr), x, y, (z + 1) % Lz, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] - res_temp[2 * Nc + j] * Complex(0, 1)) * u_temp[i * Nc + j];
      dest_temp[0 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] + res_temp[3 * Nc + j] * Complex(0, 1)) * u_temp[i * Nc + j];
      dest_temp[1 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp * Complex(0, -1);
    }
  }
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 2, x, y, (z + Lz - 1) % Lz, t, Lx, Ly, Lz, Lt, 1 - parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();

  res = getVecAddr(static_cast<Complex *>(a_ptr), x, y, (z + Lz - 1) % Lz, t, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] + res_temp[2 * Nc + j] * Complex(0, 1)) *
             u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[0 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] - res_temp[3 * Nc + j] * Complex(0, 1)) *
             u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[1 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp * Complex(0, 1);
    }
  }
  // \mu = 4
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 3, x, y, z, t, Lx, Ly, Lz, Lt, parity);
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();
  res = getVecAddr(static_cast<Complex *>(a_ptr), x, y, z, (t + 1) % Lt, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] - res_temp[2 * Nc + j]) * u_temp[i * Nc + j];
      dest_temp[0 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += -temp;
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] - res_temp[3 * Nc + j]) * u_temp[i * Nc + j];
      dest_temp[1 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += -temp;
    }
  }
  u = getGaugeAddr(static_cast<Complex *>(U_ptr), 3, x, y, z, (t + Lt - 1) % Lt, Lx, Ly, Lz, Lt, 1 - parity);
  // #pragma unroll
  // for (int i = 0; i < Nc * Nc; i++) {
  //     u_temp[i] = u[i];
  // }
  for (int i = 0; i < 2 * Nc; i++) {
    u_temp[i] = u[i];
  }
  u_temp[6] = (u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4]).conj();
  u_temp[7] = (u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5]).conj();
  u_temp[8] = (u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3]).conj();
  res = getVecAddr(static_cast<Complex *>(a_ptr), x, y, z, (t + Lt - 1) % Lt, Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    res_temp[i] = res[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (res_temp[0 * Nc + j] + res_temp[2 * Nc + j]) * u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[0 * 3 + i] += temp;
      dest_temp[2 * 3 + i] += temp;
      // second row vector with col vector
      temp = (res_temp[1 * Nc + j] + res_temp[3 * Nc + j]) * u_temp[j * Nc + i].conj(); // transpose and conj
      dest_temp[1 * 3 + i] += temp;
      dest_temp[3 * 3 + i] += temp;
    }
  }
  // end, copy result to dest
//   for (int i = 0; i < Ns * Nc; i++) {
//     dest[i] = dest_temp[i];
//   }
  // for (int i = 0; i < Ns * Nc; i++) {
  //   shared_output_vec[((threadIdx.x * Ns * Nc) << 1) + 2 * i] = dest_temp[i].real();
  //   shared_output_vec[((threadIdx.x * Ns * Nc) << 1) + 2 * i + 1] = dest_temp[i].imag();
  // }
  __syncthreads();
  // load to global memory
  for (int i = threadIdx.x; i < ((BLOCK_SIZE * Ns * Nc) << 1); i += BLOCK_SIZE) {
    shared_dest[i] = shared_output_vec[i];
  }
  __syncthreads();
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
{
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  int space = Lx * Ly * Lz * Lt >> 1;

  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  auto start = std::chrono::high_resolution_clock::now();

  // kernel function
  gpuDslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt, parity);
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);

  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
}
