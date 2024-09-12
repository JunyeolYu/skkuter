#include <torch/extension.h>
#include <torch/torch.h>
#include "ATen/ATen.h"
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>

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
    int64_t num_key_value_groups,
    torch::Tensor o_proj) {
    
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

    return torch::matmul(attn_output, o_proj.t());
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

struct Dropout_skkuter : torch::nn::Module {
    Dropout_skkuter(double prob) {
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(prob));
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        return dropout->forward(x);
    }

    torch::nn::Dropout dropout{nullptr};
};

struct Cache_skkuter {
    py::object dynamic_cache;
    // store object
    void set_dynamic_cache(py::object cache_obj) {
        dynamic_cache = cache_obj;
    }

    int64_t get_usable_length(int64_t kv_seq_len, int64_t layer_idx) {
        // py::gil_scoped_acquire acquire;
        if (dynamic_cache) {
            auto run = dynamic_cache.attr("get_usable_length")(kv_seq_len, layer_idx);
            return py::cast<int>(run);
        }
        // py::gil_scoped_release release;
    }

    std::tuple<torch::Tensor, torch::Tensor> update(torch::Tensor k, torch::Tensor v, int64_t layer_idx, py::dict args) {
        // py::gil_scoped_acquire acquire;
        if (dynamic_cache) {
            py::tuple result = dynamic_cache.attr("update")(k, v, layer_idx, args);
            return std::make_tuple(result[0].cast<torch::Tensor>(), result[1].cast<torch::Tensor>());
        }
        // py::gil_scoped_release release;
    }
};

struct Linear {
    torch::Tensor linear;
    // store object
    void set(torch::Tensor x) {
        linear = x;
    }

    torch::Tensor forward(torch::Tensor input) {
        return torch::matmul(input, linear.t());
    }
};

struct DecoderLayer {
    DecoderLayer(py::object config, int64_t layer) {
        cache = Cache_skkuter();
        layer_idx = layer;
        // Init 
        attention_dropout = config.attr("attention_dropout").cast<double>();
        hidden_size = config.attr("hidden_size").cast<int64_t>();
        num_heads = config.attr("num_attention_heads").cast<int64_t>();
        head_dim = hidden_size / num_heads;
        num_key_value_heads = config.attr("num_key_value_heads").cast<int64_t>();
        num_key_value_groups = num_heads / num_key_value_heads;
        max_position_embeddings = config.attr("max_position_embeddings").cast<int64_t>();
        original_max_position_embeddings = config.attr("original_max_position_embeddings").cast<int64_t>();
        is_causal = true;

        // init rope
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        inv_freq = 1.0 / (torch::pow(config.attr("rope_theta").cast<double>(), torch::arange(0, head_dim, 2, options) / head_dim));
    }

    torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor attention_mask, torch::Tensor position_ids, py::object past_key_value, bool output_attentions/*, bool use_cache*/) {
        // qkv_split
        auto qkv = torch::matmul(hidden_states, qkv_proj.t());
        auto bsz = qkv.size(0);
        auto q_len = qkv.size(1);
        auto pos = num_heads * head_dim;
        auto query_states = qkv.slice(-1, 0, pos);
        auto key_states = qkv.slice(-1, pos, pos + num_key_value_heads * head_dim);
        auto value_states = qkv.slice(-1, pos + num_key_value_heads * head_dim, qkv.size(-1));

        query_states = query_states.view({bsz, q_len, num_heads, head_dim}).transpose(1, 2);
        key_states = key_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);
        value_states = value_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);

        // cache 
        // Assume: `cache` is not always None and `layer_id` is given
        auto kv_seq_len = key_states.size(-2);
        cache.set_dynamic_cache(past_key_value);
        kv_seq_len += cache.get_usable_length(kv_seq_len, layer_idx);

        // rotary_embed
        auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(2).to(torch::kFloat32).expand({position_ids.size(0), -1, 1});
        auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);
        // Force float32 since bfloat16 loses precision on long contexts
        // FIXME: implement torch.autocast()
        auto freqs = torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
        auto emb = torch::cat({freqs, freqs}, -1);
        auto cos = emb.cos();
        auto sin = emb.sin();
        cos = cos.to(value_states.dtype());
        sin = sin.to(value_states.dtype());

        // apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim = 1);
        // FIXME: handling position_ids=None and unsqueeze_dim = 1
        auto cos_ = cos.unsqueeze(1); //unsqueeze_dim
        auto sin_ = sin.unsqueeze(1);
        query_states = (query_states * cos_) + (rotate_half(query_states) * sin_);
        key_states = (key_states * cos_) + (rotate_half(key_states) * sin_);

        // cache update
        // Assume: `cache` is not always None
        py::dict cache_kwargs;
        cache_kwargs["sin"] = sin;
        cache_kwargs["cos"] = cos;
        std::tuple res2 = cache.update(key_states, value_states, layer_idx, cache_kwargs);

        // attnetion forward
        // repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(std::get<0>(res2), num_key_value_groups);
        value_states = repeat_kv(std::get<1>(res2), num_key_value_groups);

        auto attn_weights = torch::matmul(query_states, key_states.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));
        attn_weights = attn_weights + attention_mask;
        attn_weights = torch::nn::functional::softmax(attn_weights, torch::nn::functional::SoftmaxFuncOptions(-1).dtype(torch::kFloat32)).to(value_states.scalar_type());
        attn_weights = torch::nn::functional::dropout(attn_weights, torch::nn::functional::DropoutFuncOptions().p(attention_dropout));

        auto attn_output = torch::matmul(attn_weights, value_states);
        attn_output = attn_output.transpose(1, 2);//.contiguous();
        attn_output = attn_output.reshape({bsz, q_len, hidden_size});
        attn_output = torch::matmul(attn_output, o_proj.t());

        // only attention output
        return attn_output;
    }

    bool set_weight (torch::Tensor qkv, torch::Tensor o) {
        qkv_proj = qkv;
        o_proj = o;

        return true;
    }

    int64_t layer_idx;
    double attention_dropout;
    int64_t hidden_size;
    int64_t num_heads;
    int64_t head_dim;
    int64_t num_key_value_heads;
    int64_t num_key_value_groups;
    int64_t max_position_embeddings;
    int64_t original_max_position_embeddings;
    bool is_causal;

    torch::Tensor qkv_proj;
    torch::Tensor o_proj;
    torch::Tensor inv_freq;
    struct Cache_skkuter cache;
};

// struct MyLinearImpl : public torch::nn::Module {
//     int64_t in_features;
//     int64_t out_features;
//     torch::Tensor weight;
//     torch::Tensor bias;
//     bool _is_hf_initialized;
//     MyLinearImpl(int64_t in_features, int64_t out_features, bool bias = true)
//         : in_features(in_features), out_features(out_features) {
//         torch::Tensor w = torch::empty({out_features, in_features}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
//         torch::nn::init::kaiming_uniform_(w, std::sqrt(5));

//         weight = register_parameter("weight", w, true);
//         // if (bias) {
//         //     this->bias = register_parameter("bias", torch::empty(out_features, torch::kFloat), true);
//         // } else {
//         // this->bias = register_parameter("bias", torch::Tensor());
//         // }
//         // reset_parameters();
//         _is_hf_initialized = true;
//     }

//     // void reset_parameters() {
//     //     torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
//     //     if (bias.defined()) {
//     //         int64_t fan_in = weight.size(1);
//     //         float bound = 1.0 / std::sqrt(fan_in);
//     //         torch::nn::init::uniform_(bias, -bound, bound);
//     //     }
//     // }

//     torch::Tensor forward(const torch::Tensor& input) {
//         return torch::linear(input, weight, bias);
//     }

//     // void load_weights(torch::Tensor new_weight, torch::Tensor new_bias) {
//     //     this->weight.copy_(new_weight);
//     //     if (this->bias.defined()) {
//     //         this->bias.copy_(new_bias);
//     //     }
//     // }

//     std::string extra_repr() {
//         std::ostringstream repr;
//         repr << "Linear(in_features=" << in_features
//              << ", out_features=" << out_features
//              << ", bias=" << (bias.defined() ? "True)" : "False)");
//         return repr.str();
//     }

//     // apply 메서드: 주어진 함수를 모든 하위 모듈에 적용
//     MyLinearImpl& apply(const std::function<void(torch::nn::Module&)>& fn) {
//         fn(*this);  // 현재 모듈에 함수 적용
//         for (auto& child : children()) {
//             child->apply(fn);  // 하위 모듈에 재귀적으로 적용
//         }
//         return *this;
//     }

//     // _get_name 메서드: 클래스 이름을 반환
//     std::string _get_name() const {
//         return "Linear";
//     }

//     // _load_from_state_dict 메서드: state_dict에서 파라미터를 로드
//     void _load_from_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
//         std::cout<<std::endl<<"_load_from_state_dict"<<std::endl;
//         if (state_dict.find("weight") != state_dict.end()) {
//             this->weight.copy_(state_dict.at("weight"));
//         }
//         if (bias.defined() && state_dict.find("bias") != state_dict.end()) {
//             this->bias.copy_(state_dict.at("bias"));
//         }
//     }

//     // state_dict를 모델에 적용하는 메서드
//     void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict, bool strict = true) {
//         // 필요한 키들을 확인
//         std::vector<std::string> missing_keys;
//         std::vector<std::string> unexpected_keys;

//         // weight 확인
//         if (state_dict.find("weight") == state_dict.end()) {
//             missing_keys.push_back("weight");
//         }
//         // bias 확인
//         if (bias.defined() && state_dict.find("bias") == state_dict.end()) {
//             missing_keys.push_back("bias");
//         }

//         // strict 모드일 때 예상치 못한 키가 있으면 오류 처리
//         for (const auto& kv : state_dict) {
//             if (kv.first != "weight" && kv.first != "bias") {
//                 unexpected_keys.push_back(kv.first);
//             }
//         }

//         if (strict && (!missing_keys.empty() || !unexpected_keys.empty())) {
//             throw std::runtime_error("Missing or unexpected keys in state_dict.");
//         }

//         // 실제로 state_dict에서 로드
//         _load_from_state_dict(state_dict);
//     }

//     torch::Tensor get_weight() {
//         return weight;
//     }
// };
// TORCH_MODULE(Linear);

class MyModuleImpl : public torch::nn::Module {
public:
    MyModuleImpl(int64_t in_features, int64_t out_features, bool bias = true) {
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }
private:
    torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(MyModule);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_kv", &repeat_kv, "repeat_kv");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "apply_rotary_pos_emb");
    m.def("attention_forward", &attention_forward, "Attention forward pass in C++");
    m.def("RMSnorm_forward", &RMSnorm_forward, "RMSnorm_forward");
    m.def("qkv_split", &qkv_split, "qkv_split");
    py::class_<Phi3RotaryEmbedding>(m, "Phi3RotaryEmbedding")
        .def(py::init<int64_t, int64_t, double>())
        .def("forward", &Phi3RotaryEmbedding::forward);
    py::class_<Dropout_skkuter, torch::nn::Module, std::shared_ptr<Dropout_skkuter>>(m, "Dropout_skkuter")
        .def(py::init<double>())
        .def("__call__", &Dropout_skkuter::forward)
        .def("forward", &Dropout_skkuter::forward);
    py::class_<Cache_skkuter, std::shared_ptr<Cache_skkuter>>(m, "Cache_skkuter")
        .def(py::init<>())
        .def("set_dynamic_cache", &Cache_skkuter::set_dynamic_cache)
        .def("get_usable_length", &Cache_skkuter::get_usable_length)
        .def("update", &Cache_skkuter::update);
    py::class_<Linear, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<>())
        .def("__call__", &Linear::forward)
        .def("set", &Linear::set)
        .def("forward", &Linear::forward);
    py::class_<DecoderLayer, std::shared_ptr<DecoderLayer>>(m, "DecoderLayer")
        .def(py::init<py::object, int64_t>())
        .def("__call__", &DecoderLayer::forward)
        .def("forward", &DecoderLayer::forward)
        .def("set_weight", &DecoderLayer::set_weight);
    // py::class_<MyLinearImpl, torch::nn::Module, std::shared_ptr<MyLinearImpl>>(m, "Linear")
    //     .def(py::init<int64_t, int64_t, bool>())
    //     .def("__call__", &MyLinearImpl::forward)
    //     .def("forward", &MyLinearImpl::forward)
    //     .def("__repr__", &MyLinearImpl::extra_repr)
    //     .def("extra_repr", &MyLinearImpl::extra_repr)
    //     .def("apply", &MyLinearImpl::apply)
    //     .def("_get_name", &MyLinearImpl::_get_name)  // _get_name 메서드 바인딩
    //     .def("_load_from_state_dict", &MyLinearImpl::_load_from_state_dict)  // _load_from_state_dict 바인딩
    //     .def("load_state_dict", &MyLinearImpl::load_state_dict)  // _load_from_state_dict 바인딩
    //     .def("get_weight", &MyLinearImpl::get_weight)
    //     // .def("load_weights", &MyLinearImpl::load_weights)
    //     .def_readwrite("in_features", &MyLinearImpl::in_features)
    //     .def_readwrite("out_features", &MyLinearImpl::out_features)
    //     .def_readwrite("weight", &MyLinearImpl::weight)
    //     .def_readwrite("bias", &MyLinearImpl::bias)
    //     .def_readwrite("_is_hf_initialized", &MyLinearImpl::_is_hf_initialized);
    py::class_<MyModuleImpl, torch::nn::Module, std::shared_ptr<MyModuleImpl>>(m, "MyModule")
        .def(py::init<int64_t, int64_t, bool>())
        .def("__call__", &MyModuleImpl::forward)
        .def("forward", &MyModuleImpl::forward);
}