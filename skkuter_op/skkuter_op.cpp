#include <torch/extension.h>
#include <torch/torch.h>
#include "ATen/ATen.h"
#include <cmath>
#include <vector>

torch::Tensor repeat_kv(torch::Tensor hidden_states, int64_t n_rep);

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
    int64_t hidden_size,
    int64_t num_key_value_groups) {
    
    // repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, num_key_value_groups);
    value_states = repeat_kv(value_states, num_key_value_groups);

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
    attn_weights = torch::nn::functional::dropout(attn_weights, torch::nn::functional::DropoutFuncOptions().p(attention_dropout));

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

    attn_output = attn_output.transpose(1, 2);//.contiguous();
    attn_output = attn_output.reshape({bsz, q_len, hidden_size});

    return attn_output;
}

torch::Tensor repeat_kv(torch::Tensor hidden_states, int64_t n_rep) {
    //This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    //num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    auto size =  hidden_states.sizes();
    auto batch = size[0];
    auto num_key_value_heads = size[1];
    auto slen = size[2];
    auto head_dim = size[3];

    if (n_rep == 1) return hidden_states;
    hidden_states = hidden_states.unsqueeze(2).expand({batch, num_key_value_heads, n_rep, slen, head_dim});
    return hidden_states.reshape({batch, num_key_value_heads * n_rep, slen, head_dim});
}

torch::Tensor rotate_half(torch::Tensor x) {
    auto half = x.size(-1) / 2;
    return torch::cat({-x.slice(-1, half), x.slice(-1, 0, half)}, -1);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::optional<torch::Tensor> position_id,
    int64_t unsqueeze_dim = 1) {
    
    cos = cos.unsqueeze(unsqueeze_dim);
    sin = sin.unsqueeze(unsqueeze_dim);
    auto q_embed = (q * cos) + (rotate_half(q) * sin);
    auto k_embed = (k * cos) + (rotate_half(k) * sin);
    return std::make_tuple(q_embed, k_embed);
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

torch::Tensor RMSnorm_forward(torch::Tensor hidden_states, double eps) {
    auto input_dtype = hidden_states.scalar_type();
    hidden_states = hidden_states.to(torch::kFloat32);
    auto variance = hidden_states.pow(2).mean(-1, true);
    hidden_states = hidden_states * torch::rsqrt(variance + eps);
    return hidden_states.to(input_dtype);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qkv_split(
    torch::Tensor qkv, int64_t num_heads, int64_t head_dim, int64_t num_key_value_heads) {

    auto bsz = qkv.size(0);
    auto q_len = qkv.size(1);
    auto pos = num_heads * head_dim;
    auto query_states = qkv.slice(-1, 0, pos);
    auto key_states = qkv.slice(-1, pos, pos + num_key_value_heads * head_dim);
    auto value_states = qkv.slice(-1, pos + num_key_value_heads * head_dim, qkv.size(-1));

    query_states = query_states.view({bsz, q_len, num_heads, head_dim}).transpose(1, 2);
    key_states = key_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);
    value_states = value_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);

    return std::make_tuple(query_states, key_states, value_states);
}

struct Linear_skkuter : torch::nn::Module {
    Linear_skkuter(torch::Tensor weight) {
        linear = torch::nn::Linear(torch::nn::LinearOptions(weight.size(0), weight.size(1)).bias(false));
        register_module("linear", linear);
        linear->weight = weight.clone(); 
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }

    torch::nn::Linear linear{nullptr};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_kv", &repeat_kv, "repeat_kv");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "apply_rotary_pos_emb");
    m.def("attention_forward", &attention_forward, "Attention forward pass in C++");
    m.def("RMSnorm_forward", &RMSnorm_forward, "RMSnorm_forward");
    m.def("qkv_split", &qkv_split, "qkv_split");
    py::class_<Phi3RotaryEmbedding>(m, "Phi3RotaryEmbedding")
        .def(py::init<int64_t, int64_t, double>())
        .def("forward", &Phi3RotaryEmbedding::forward);
    py::class_<Linear_skkuter, torch::nn::Module, std::shared_ptr<Linear_skkuter>>(m, "Linear_skkuter")
        .def(py::init<torch::Tensor>())
        .def("__call__", &Linear_skkuter::forward)
        .def("forward", &Linear_skkuter::forward);
}