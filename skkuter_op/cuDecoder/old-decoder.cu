// #include <torch/types.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include<inttypes.h>

// #include <cublas_v2.h>
// #include <cuda_bf16.h>  // For __nv_bfloat16
// #include <ATen/cuda/CUDAContext.h>  // For getting the CUDA stream
// #include <vector>

// #include <cutlass/cutlass.h>
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/array.h>





// __global__
// void rms_norm_forward_kernel(float* hidden_states, float* rms_norm_weight, float* rsqrt, int seq_len, int N) 
// {

//     int current_batch = blockIdx.z;
//     int current_seq = blockIdx.y;

//     float* my_hidden_states = hidden_states + (current_seq * N) + (current_batch * seq_len * N);
//     float* my_rsqrt = rsqrt + (current_seq * 1) + (current_batch * seq_len * 1);

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     if(tid < N)
//     {
//         my_hidden_states[tid] = my_hidden_states[tid] * (*my_rsqrt) * rms_norm_weight[tid];
//     }
// }



// torch::Tensor batched_bfloat16_gemm(torch::Tensor B, torch::Tensor A) {
//     // Ensure tensors are on CUDA device and of type bfloat16
//     if (!A.is_cuda() || !B.is_cuda() || A.scalar_type() != torch::kFloat32 || B.scalar_type() != torch::kFloat32) {
//         throw std::runtime_error("Tensors A and B must be on CUDA device and of type torch::kBFloat16");
//     }

//     // Extract dimensions
//     /*
//         B is hidden state of shape              => (batch_size, hidden_size, seq_len)
//         A is qkv_weight of shape                => (proj_dim, hidden_size)
//         C = A * B
//         C is of shape                           => (batch_size, proj_dim, seq_len)
//     */
//     int64_t batch_size = B.size(0);
//     int64_t m = A.size(0);
//     int64_t k = A.size(1);
//     int64_t n = B.size(2);

//     if (A.size(1) != B.size(1)) {
//         throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
//     }


//     torch::Tensor C = torch::empty({batch_size, m, n}).to(A.device()).to(torch::kFloat32);

//     // fill C with zeros
//     C = C.fill_(0);
//     // Get data pointers
//     float* A_device = A.data_ptr<float>();
//     float* B_device = B.data_ptr<float>();
//     float* C_device = C.data_ptr<float>();

//     // Create cuBLAS handle and set stream
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
//     cublasSetStream(handle, stream);

//     // Set scalar values for multiplication
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Set data types
//     cudaDataType_t Atype = CUDA_R_32F;
//     cudaDataType_t Btype = CUDA_R_32F;
//     cudaDataType_t Ctype = CUDA_R_32F;
//     cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

//     //A is row major matrix so its lda is the number of rows
//     // Perform batched GEMM
//     cublasStatus_t status = cublasGemmStridedBatchedEx(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         m, n, k,
//         &alpha,
//         A_device, Atype, m, 0,   // lda is k, stride for A is m * k
//         B_device, Btype, k, k * n,
//         &beta,
//         C_device, Ctype, m, m * n,  // ldc is n, stride for C is m * n
//         batch_size,
//         computeType,
//         CUBLAS_GEMM_DEFAULT_TENSOR_OP);

//     if (status != CUBLAS_STATUS_SUCCESS) {
//         cublasDestroy(handle);
//         throw std::runtime_error("cuBLAS GEMM operation failed with status: " + std::to_string(status));
//     }

//     // Destroy cuBLAS handle
//     cublasDestroy(handle);

//     return C.transpose(1, 2);
// }


// torch::Tensor cuda_attn_forward(
//     /* First RMSNorm Layer*/torch::Tensor hidden_states,  double rms_norm_epsilon, torch::Tensor rms_norm_weight,
//     /* QKV Projection     */torch::Tensor qkv_weight
//     ) 
// {

//     /*
//         Specs of our machine
//             GPU: Jetson AGX Orin 32 GB
//             CUDA: 12.*

//             SRAM size: 192KB
//             SM: 14
//             Cores/SM: 128
//             TensorCores/SM: 4
//     */

//     /********************************************* RMSNorm *********************************************/
//     // torch::NoGradGuard no_grad;
//     // auto input_dtype = hidden_states.scalar_type();
//     // hidden_states = hidden_states.to(torch::kFloat32);
//     // float* hidden_states_pointer = hidden_states.data_ptr<float>();

//     // rms_norm_weight = rms_norm_weight.to(torch::kFloat32);
//     // float* rms_norm_weight_pointer = rms_norm_weight.data_ptr<float>();

//     // auto rsqrt = torch::rsqrt(hidden_states.pow(2).mean(-1, true) + rms_norm_epsilon);
//     // float* rsqrt_pointer = rsqrt.data_ptr<float>();

//     // int batch_size = hidden_states.size(0);
//     // int seq_len = hidden_states.size(1);
//     // int hidden_size = hidden_states.size(2);

//     // dim3 threads_per_block(512,1,1);
//     // dim3 number_of_blocks(10,seq_len,batch_size);
    

//     // rms_norm_forward_kernel<<<number_of_blocks, threads_per_block>>>(
//     //     hidden_states_pointer, rms_norm_weight_pointer,
//     //     rsqrt_pointer, 
//     //     seq_len, hidden_size
//     // );

//     // cudaDeviceSynchronize();

//     // /********************************************* QKV Projection *********************************************/
//     // hidden_states = hidden_states.to(input_dtype);
//     // return torch::matmul(hidden_states, qkv_weight.t());

// }






// #include <torch/types.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <torch/extension.h>

// #include <torch/torch.h>
// #include "ATen/ATen.h"



// #define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
// void check(cudaError_t err, const char* const func, const char* const file,
//            const int line)
// {
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA Runtime Error at: " << file << ":" << line
//                   << std::endl;
//         std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
//         // We don't exit when we encounter CUDA errors in this example.
//         // std::exit(EXIT_FAILURE);
//     }
// }

// #define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
// void checkLast(const char* const file, const int line)
// {
//     cudaError_t const err{cudaGetLastError()};
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA Runtime Error at: " << file << ":" << line
//                   << std::endl;
//         std::cerr << cudaGetErrorString(err) << std::endl;
//         // We don't exit when we encounter CUDA errors in this example.
//         // std::exit(EXIT_FAILURE);
//     }
// }



// __global__
// void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int kN, const int d,
//                     const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
//                     float* l, float *m, float* O, float *mask) {
//     int tx = threadIdx.x;
//     int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

//     // Offset into Q,K,V,O,l,m - different for each batch and head
//     int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
//     int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

//     // Define SRAM for Q,K,V,S
//     extern __shared__ float sram[];
//     int tile_size = Bc * d;  // size of Qi, Kj, Vj
//     float* Qi = sram;
//     float* Kj = &sram[tile_size];
//     float* Vj = &sram[tile_size * 2];
//     float* S = &sram[tile_size * 3];

    



//     for (int j = 0; j < Tc; j++) {

//         // if the tx is less than the remaining columns in the last block continue the loop otherwise break
//         if(j == Tc-1 && (tx > (N%Bc - 1))) break;


//         // Load Kj, Vj to SRAM
//         for (int x = 0; x < d; x++) {
//             Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
//             Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
//         }
//         __syncthreads();  // such that the inner loop can use the correct Kj, Vj

//         for (int i = 0; i < Tr; i++)  {

//             // Load Qi to SRAM, l and m to registers
//             for (int x = 0; x < d; x++) {
//                 Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
//             }
//             float row_m_prev = m[lm_offset + (Br * i) + tx];
//             float row_l_prev = l[lm_offset + (Br * i) + tx];

//             // S = QK^T, row_m = rowmax(S)
//             float row_m = -INFINITY;
//             for (int y = 0; y < Bc; y++) {
//                 float sum = 0;
//                 for (int x = 0; x < d; x++) {
//                     sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
//                 }
//                 sum *= softmax_scale;

                
//                 // int mask_offset = (bx * N * kN) + (i * Bc * kN) + (tx * kN) + (j * Bc) + y;
//                 S[(Bc * tx) + y] = sum; //+ mask[mask_offset];

//                 if (sum > row_m)
//                     row_m = sum;
//             }



//             // P = exp(S - row_m), row_l = rowsum(P)
//             float row_l = 0;
//             for (int y = 0; y < Bc; y++) {
//                 S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
//                 row_l += S[(Bc * tx) + y];
//             }

//             //implement the dropout in here

//             // Compute new m and l
//             float row_m_new = max(row_m_prev, row_m);
//             float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

//             // Write O, l, m to HBM
//             for (int x = 0; x < d; x++) {
//                 float pv = 0;  // Pij * Vj
//                 for (int y = 0; y < Bc; y++) {
//                     pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
//                 }
//                 O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
//                     * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
//                     + (__expf(row_m - row_m_new) * pv));
//             }

//             m[lm_offset + (Br * i) + tx] = row_m_new;
//             l[lm_offset + (Br * i) + tx] = row_l_new;
//         }
//         __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
//     }
// }




// torch::Tensor myTest() {


//     //print them

//     auto input_type = Q.scalar_type();

//     Q = Q.to(torch::kFloat32);
//     K = K.to(torch::kFloat32);
//     V = V.to(torch::kFloat32);
//     mask = mask.to(torch::kFloat32);
    

//     //masking is done only in the beginning of the forward pass

//     // TODO: determine Bc, Br dynamically
//     const int Bc = 16; const int Br = 16;

//     const int B = Q.size(0); const int nh = Q.size(1);
//     const int N = Q.size(2); const int d = Q.size(3);
//     const int kN = K.size(2);


//     int Tc = ceil((float) N / Bc); int Tr = ceil((float) N / Br);
//     //Tc and Tr must be greater than 0
//     Tc = max(Tc, 1);
//     Tr = max(Tr, 1);


//     const float softmax_scale = 1.0 / sqrt(d);

//     // Initialize O, l, m to HBM
//     auto O = torch::zeros_like(Q);
    

//     auto l = torch::zeros({B, nh, N});
//     auto m = torch::full({B, nh, N}, -INFINITY);
//     torch::Device device(torch::kCUDA);
//     l = l.to(device); m = m.to(device);

//     // Calculate SRAM size needed per block
//     const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
//     int max_sram_size;
//     cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

//     dim3 grid_dim(B, nh);  // batch_size x num_heads
//     dim3 block_dim(Bc);  // Bc threads per block

//     //print O shape and mask shape
//     // printf("O shape: %d %d %d %d\n", O.size(0), O.size(1), O.size(2), O.size(3));
//     // printf("Mask shape: %d %d %d %d\n", mask.size(0), mask.size(1), mask.size(2), mask.size(3));

//     forward_kernel<<<grid_dim, block_dim, sram_size>>>(
//         Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
//         N,kN, d, Tc, Tr, Bc, Br, softmax_scale,
//         l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>(), mask.data_ptr<float>());

//     // printf("Done with the kernel\n");
//     //weight for the operation to finish
//     cudaDeviceSynchronize();

//     CHECK_LAST_CUDA_ERROR();


//     return O.to(input_type);
// }

// '
//     /* KHAN
//         A single block should be of Br x Bc size
//         Br = min(M/4d, qN)
//         Bc = min(M/4d, kN)

//         Number of tiles of size Br x Bc will be
//         Tr = ceil(qN/Br)
//         Tc = ceil(kN/Bc)
//     */
//     int Br = std::min(M/(4*d), qN); 
//     int Bc = std::min(M/(4*d), kN);

//     int Tr = (qN + Br - 1)/Br; 
//     int Tc = (kN + Bc - 1)/Bc;
