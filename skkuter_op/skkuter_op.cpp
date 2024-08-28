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
    int64_t hidden_size) {

    // calculate attention weights
    auto attn_weights = torch::matmul(query_states, key_states.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));

    // check attention weights size
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

    // apply attention mask if provided
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

    // upcast to fp32
    attn_weights = torch::nn::functional::softmax(attn_weights, torch::nn::functional::SoftmaxFuncOptions(-1).dtype(torch::kFloat32)).to(value_states.scalar_type());
    attn_weights = torch::nn::functional::dropout(attn_weights, torch::nn::functional::DropoutFuncOptions().p(attention_dropout).training(false));

    // calculate attention output
    auto attn_output = torch::matmul(attn_weights, value_states);

    // check attention output size
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
    attn_output = attn_output.reshape({bsz, q_len, hidden_size});

    return attn_output;
}

class Phi3RotaryEmbedding {
public:
    Phi3RotaryEmbedding(int64_t dim, int64_t max_position_embeddings, double base)
        : dim(dim), max_position_embeddings(max_position_embeddings), base(base) {
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA); // 
        inv_freq = 1.0 / (torch::pow(base, torch::arange(0, dim, 2, options) / dim));
    }

    std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor position_ids) {
        // x: [bs, num_attention_heads, seq_len, head_size]
        auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(2).to(torch::kFloat32).expand({position_ids.size(0), -1, 1});
        auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);
        // Force float32 since bfloat16 loses precision on long contexts
        // FIXME: implement torch.autocast()
        auto freqs = torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
        auto emb = torch::cat({freqs, freqs}, -1);
        auto cos = emb.cos();
        auto sin = emb.sin();

        return {cos.to(x.dtype()), sin.to(x.dtype())};
    }

private:
    int64_t dim;
    int64_t max_position_embeddings;
    double base;
    torch::Tensor inv_freq;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention forward pass in C++");
    py::class_<Phi3RotaryEmbedding>(m, "Phi3RotaryEmbedding")
        .def(py::init<int64_t, int64_t, double>())
        .def("forward", &Phi3RotaryEmbedding::forward);
}