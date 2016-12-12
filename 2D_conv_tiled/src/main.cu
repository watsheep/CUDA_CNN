
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_WIDTH 8


static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}
/*
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, filter_h)) {
            for (const auto q : range(0, filter_w)) {
              for (const auto c : range(0, in_channel)) {
                const auto yoffset =
                    ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}
*/
// From book chapter Figure 16.4
__global__ void conv_forward_valid_kernel(float *X, int batch,int H_in,int W_in,int Channel_in,
                                         int H_out,int W_out,int M_out, float *Y,
                                         int filter_h,int filter_w,
                                        float *W,int W_grid) {
  int n=blockIdx.x;
  int m = blockIdx.y;
  int h_base = (blockIdx.z / W_grid)*TILE_WIDTH ; 
  int w_base = (blockIdx.z % W_grid)*TILE_WIDTH ;
  int h0 = threadIdx.y;
  int w0 = threadIdx.x; 
  int h = h_base+h0;
  int w = w_base+w0;
  int X_tile_width = TILE_WIDTH + filter_w-1; 
  extern __shared__ float shared[];
  float* x_shared = &shared[0];
  float* w_shared = &shared[X_tile_width*X_tile_width];
 
  if(h<H_out&&w<W_out){
    float tmp = 0.0;
    int yoffset = ((n * H_out + h) * W_out + w) * M_out + m;
    for (const auto c : range(0, Channel_in)) {
      if(h0<filter_h&&w0<filter_w){
        const auto woffset = h0 * filter_w * Channel_in * M_out +
                                       w0 * Channel_in * M_out + c * M_out + m;
        w_shared[h0*filter_h+w0]=W[woffset];
      }
      __syncthreads();
      for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) { 
        for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
          const auto xoffset = n * H_in * W_in * Channel_in +
                                       (i) * W_in * Channel_in +
                                       (j) * Channel_in + c;
          x_shared[(i - h_base)*X_tile_width+j - w_base] = X[xoffset]; 
         }
       }
       __syncthreads();
      for (const auto p : range(0, filter_h)) {
        for (const auto q : range(0, filter_w)) {  
          tmp += x_shared[(h0+p)*X_tile_width+w0+q] * w_shared[p*filter_h+q];    
          }
        }
      __syncthreads();
      }
      Y[yoffset] = tmp;
   }
  
  }
 
      
// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset =
                  ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  
  //change here. invoke the kernel
  int con1W_grid = ceil(adims[2]/TILE_WIDTH); // number of horizontal tiles per output map 
  int con1H_grid = ceil(adims[1]/TILE_WIDTH); // number of vertical tiles per output map
  int sizeOfOutput = adims[0]*adims[1]*adims[2]*adims[3];
  int sizeOfX = xdims[0]*xdims[1]*xdims[2]*xdims[3];
  int con1Z = con1H_grid * con1W_grid;
  size_t shmem_size = sizeof(float) *((TILE_WIDTH+conv1dims[0]+1)*(TILE_WIDTH+conv1dims[0]+1)+conv1dims[0]*conv1dims[0]);
  //TODO cuda malloc
  
  float *deviceX;
  float *deviceA;
  float *convW1;
  cudaMalloc((void **) &deviceX,sizeOfX * sizeof(float));
  cudaMalloc((void **) &deviceA,sizeOfOutput * sizeof(float));
  cudaMalloc((void **) &convW1,800 * sizeof(float));
  //TODO cudamcpy
  cudaMemcpy(deviceX,x,sizeOfX * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(convW1,conv1,800 * sizeof(float),cudaMemcpyHostToDevice);
  //call kerenl
  dim3 con_blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 con_gridDim(adims[0], adims[3], con1Z);
  conv_forward_valid_kernel<<<con_gridDim,con_blockDim,shmem_size>>>(deviceX, xdims[0], xdims[1],xdims[2],xdims[3],
                                                             adims[1],adims[2],adims[3],deviceA,
                                                             conv1dims[0],conv1dims[1],convW1,
                                                             con1W_grid);
  cudaDeviceSynchronize();

  cudaMemcpy(a,deviceA,sizeOfOutput * sizeof(float),cudaMemcpyDeviceToHost);
  
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  
  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);

  int con2W_grid = ceil(cdims[2]/TILE_WIDTH);
  int con2H_grid = ceil(cdims[1]/TILE_WIDTH);
  int con2Z = con2H_grid*con2W_grid;
  int sizeOfc = cdims[0]*cdims[1]*cdims[2]*cdims[3];
  int sizeOfb = bdims[0]*bdims[1]*bdims[2]*bdims[3];
  size_t shmem_size2 = sizeof(float) *((TILE_WIDTH+conv2dims[0]+1)*(TILE_WIDTH+conv2dims[0]+1)+conv2dims[0]*conv2dims[0]);
  //TODO cuda malloc
  float *deviceB;
  float *deviceC;
  float *convW2;
  //TODO malloc
  cudaMalloc((void **) &deviceB,sizeOfb * sizeof(float));
  cudaMalloc((void **) &deviceC,sizeOfc * sizeof(float));
  cudaMalloc((void **) &convW2,51200 * sizeof(float));
  //TODO cudamemcpy
  cudaMemcpy(deviceB,b,sizeOfb * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(convW2,conv2,51200 * sizeof(float),cudaMemcpyHostToDevice);
  dim3 con_blockDim2(TILE_WIDTH,TILE_WIDTH,1);
  dim3 con_gridDim2(cdims[0], cdims[3], con2Z);
  conv_forward_valid_kernel<<<con_gridDim2,con_blockDim2,shmem_size2>>>(deviceB, bdims[0], bdims[1],bdims[2],bdims[3],
                                                             cdims[1],cdims[2],cdims[3],deviceC,
                                                             conv2dims[0],conv2dims[1],convW2,
                                                             con2W_grid);
  //TODO copy memory back
  cudaDeviceSynchronize();
  cudaMemcpy(c,deviceC,sizeOfc * sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  //conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);
  // relu
  
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
