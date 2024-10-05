#include <torch/types.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <torch/torch.h>
#include "ATen/ATen.h"
#include <cmath>

#define ELEMENT_TYPE torch::kBFloat16


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}



/*
    KHAN

    This kernel is responsible for carrying out the attention between 1 query and all the keys
    The kernel will first multiply Q and transposed_k 
    Then it will scale the dot product
    Then it will apply the softmax
    Then it will multiply the softmax with the values
    Finally it will store the result in the output tensor

*/


#define CONVERT(x) __float2bfloat16(x)
#define CONVERT_TO_FLOAT(x) __bfloat162float(x)
#define BTYPE __nv_bfloat16

__global__
void attention_forward_kernel(BTYPE* Q, BTYPE* K, BTYPE* V, BTYPE* O, BTYPE* mask,
                                float div,
                                int Tc, int Bc,
                                int d, int qN, int kN){

    int batch = gridDim.z;
    int nH = gridDim.y;
    int tx = threadIdx.x;

    int current_batch = blockIdx.z;
    int current_head = blockIdx.y;
    int current_seq = blockIdx.x;

    
    int q_offset = (current_batch * nH * qN * d) + (current_head * qN * d) + (current_seq * d);  
    int o_offset = (current_batch * nH * qN * d) + (current_head * qN * d) + (current_seq * d);  
    int k_offset = (current_batch * nH * kN * d) + (current_head * kN * d);
    int v_offset = (current_batch * nH * kN * d) + (current_head * kN * d);
    int mask_offset = (current_batch * kN * qN) + (current_seq * kN);

    BTYPE* q = Q + q_offset; // (1 x d)
    BTYPE* o = O + o_offset; // (1 x d)
    BTYPE* v = V + v_offset; // (kN x d)
    BTYPE* k = K + k_offset; // (d x kN)

    extern __shared__ float sram[];
    float* shared_o = sram;

    
    for(int i = 0; i < Tc; i++){
        BTYPE* k_ptr = k + (i * Bc); // (d x Bc)
        float sum = 0.0f;

        if(i*Bc + tx >= kN){
            continue;
        }

        for(int j = 0; j < d; j++){
            sum += CONVERT_TO_FLOAT(q[j] * k_ptr[j * kN + tx]);
        }

        shared_o[i*Bc+tx] = sum * div;
        shared_o[i*Bc+tx] += CONVERT_TO_FLOAT(mask[mask_offset + i*Bc+tx]);
        
    }

    __syncthreads();

    //Simply compute the softmax
    float sum = 0.0f;
    float max = -INFINITY;
    for(int i = 0; i < kN; i++){;
        if(shared_o[i] > max){
            max = shared_o[i];
        }
    }

    for(int i = tx; i < kN; i = i + Bc){
        if(i < kN)
            shared_o[i] = __expf(shared_o[i] - max);
    }

    for(int i = 0; i < kN; i++){
        sum += shared_o[i];        
    }

    for(int i = tx; i < kN; i = i + Bc){
        if(i < kN)
            shared_o[i] /= sum;
    }

    __syncthreads();

    if(tx == 0){
        for(int i = 0; i < d; i++){
            float sum = 0.0f;
            for(int j = 0; j < kN; j++){
                sum += shared_o[j] * CONVERT_TO_FLOAT(v[j * d + i]);
            }

            o[i] = CONVERT(sum);
        }

    }

}

int get_sram_size(){
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.sharedMemPerMultiprocessor;

}

torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask){

    K = K.transpose(2,3);

    int M = get_sram_size();
    int batch = Q.size(0); int nH = Q.size(1); 
    int qN = Q.size(2); int kN = K.size(3); //Because k is transposed
    int d = Q.size(3);

    //For scaling the dot product
    auto div = 1.0f/std::sqrt(static_cast<float>(d));


    /* KHAN
        The number of columns in K will determine the number of threads
        But it should be multiple of 32
        Get bc such that kN is a multiple of bc
    */

    int Bc = 32; 
    int Tc = ceil((float)kN / (float)Bc);
    
    dim3 block(Bc, 1, 1);
    dim3 grid(qN,nH, batch);

    /* KHAN
        Create an O tensor of size batch x nH x qN x d
        Create the pointers for the Q, K, V and O tensors
    */

    torch::Tensor O = torch::zeros({batch, nH, qN, d}, Q.options());

    __nv_bfloat16* Q_ptr = reinterpret_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = reinterpret_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = reinterpret_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = reinterpret_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* mask_ptr = reinterpret_cast<__nv_bfloat16*>(mask.data_ptr());


    /* KHAN
        Launch the kernel for attention
    */
    attention_forward_kernel<<<grid, block, kN*4>>>(Q_ptr, K_ptr, V_ptr, O_ptr, mask_ptr,
        div,
        Tc, Bc,
        d, qN, kN);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    return O;

    


}


void myTest(){
    torch::Tensor Q = torch::randn({1, 40, 1, 128}, ELEMENT_TYPE).to(torch::kCUDA);
    torch::Tensor Q2 = torch::empty({1, 40, 1, 128}, ELEMENT_TYPE).to(torch::kCUDA);

    torch::Tensor K = torch::randn({1, 40, 170, 128}, ELEMENT_TYPE).to(torch::kCUDA);
    torch::Tensor K2 = torch::empty({1, 40, 170, 128}, ELEMENT_TYPE).to(torch::kCUDA);
    
    torch::Tensor V = torch::randn({1, 40, 170, 128}, ELEMENT_TYPE).to(torch::kCUDA);
    torch::Tensor V2 = torch::empty({1, 40, 170, 128}, ELEMENT_TYPE).to(torch::kCUDA);

    torch::Tensor mask = torch::randn({1, 1, 1, 170}, ELEMENT_TYPE).to(torch::kCUDA);
    mask.fill_(1);
    auto O = attention_forward(Q, K, V, mask);

    Q2.copy_(Q);
    K2.copy_(K);
    V2.copy_(V);



    // //do the attention by pytorch
    auto div = std::sqrt(128);
    auto query_states = torch::matmul(Q2, K2.transpose(2,3)) / div; 
    query_states = query_states + mask;
    query_states = torch::nn::functional::softmax(query_states, torch::nn::functional::SoftmaxFuncOptions(-1.f)).to(ELEMENT_TYPE);
    auto value_states = torch::matmul(query_states, V2);

    //get the different between O and value_states
    auto diff = torch::abs(value_states - O);

    // //print the max diff
    std::cout << "Max diff: " << diff.max().item<float>() << std::endl;

}
