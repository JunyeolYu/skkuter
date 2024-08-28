#include <torch/extension.h>
#include "ATen/ATen.h"
#include <cmath>
#include <vector>

torch::Tensor attention_forward(
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor attention_mask,
    int64_t head_dim,
    int64_t bsz,
    int64_t num_heads,
    int64_t q_len,
    int64_t kv_seq_len,
    double attention_dropout,
    bool training) {

    // Calculate attention weights
    auto attn_weights = torch::matmul(query_states, key_states.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));

    // Check attention weights size
    if (attn_weights.sizes() != std::vector<int64_t>{bsz, num_heads, q_len, kv_seq_len}) {
        throw std::runtime_error("Attention weights should be of size (" + 
                                  std::to_string(bsz) + ", " + 
                                  std::to_string(num_heads) + ", " + 
                                  std::to_string(q_len) + ", " + 
                                  std::to_string(kv_seq_len) + "), but got " +
                                  std::to_string(attn_weights.size(0)) + ", " + 
                                  std::to_string(attn_weights.size(1)) + ", " + 
                                  std::to_string(attn_weights.size(2)) + ", " + 
                                  std::to_string(attn_weights.size(3)));
    }

    // Apply attention mask if provided
    if (attention_mask.defined()) {
        if (attention_mask.sizes() != std::vector<int64_t>{bsz, 1, q_len, kv_seq_len}) {
            throw std::runtime_error("Attention mask should be of size (" + 
                                      std::to_string(bsz) + ", 1, " + 
                                      std::to_string(q_len) + ", " + 
                                      std::to_string(kv_seq_len) + "), but got " + 
                                      std::to_string(attention_mask.size(0)) + ", " + 
                                      std::to_string(attention_mask.size(1)) + ", " + 
                                      std::to_string(attention_mask.size(2)) + ", " + 
                                      std::to_string(attention_mask.size(3)));
        }
        attn_weights = attn_weights + attention_mask;
    }

    // Upcast to fp32 and apply softmax
    attn_weights = torch::nn::functional::softmax(attn_weights, torch::nn::functional::SoftmaxFuncOptions(-1).dtype(torch::kFloat32)).to(value_states.scalar_type());

    // Apply dropout
    attn_weights = torch::nn::functional::dropout(attn_weights, torch::nn::functional::DropoutFuncOptions().p(attention_dropout).training(training));

    // Calculate attention output
    auto attn_output = torch::matmul(attn_weights, value_states);

    // Check attention output size
    if (attn_output.sizes() != std::vector<int64_t>{bsz, num_heads, q_len, head_dim}) {
        throw std::runtime_error("`attn_output` should be of size (" + 
                                  std::to_string(bsz) + ", " + 
                                  std::to_string(num_heads) + ", " + 
                                  std::to_string(q_len) + ", " + 
                                  std::to_string(head_dim) + "), but got " + 
                                  std::to_string(attn_output.size(0)) + ", " + 
                                  std::to_string(attn_output.size(1)) + ", " + 
                                  std::to_string(attn_output.size(2)) + ", " + 
                                  std::to_string(attn_output.size(3)));
    }

    attn_output = attn_output.transpose(1, 2).contiguous();
    // attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    return attn_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention forward pass in C++");
}