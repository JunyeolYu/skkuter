#include "skkuter_op.h"

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

torch::Tensor RMSnorm_forward(torch::Tensor hidden_states, double eps) {
    torch::NoGradGuard no_grad;
    auto input_dtype = hidden_states.scalar_type();
    hidden_states = hidden_states.to(torch::kFloat32);
    hidden_states = hidden_states * torch::rsqrt(hidden_states.pow(2).mean(-1, true) + eps);
    return hidden_states.to(input_dtype);
}

struct Cache {
    Cache(int64_t batch_size, int64_t seq_len, int64_t num_layers) {
        current_length = 0;
        max_seq_length = seq_len + 25; // FIXME: this should be changed
        auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        for (int i = 0; i < num_layers; i++) {
            key_cache.push_back(torch::zeros({batch_size, 10, max_seq_length, 128}, options));
            value_cache.push_back(torch::zeros({batch_size, 10, max_seq_length, 128}, options));
        }
    }

    int64_t get_seq_length(int64_t layer_idx = 0) {
        if ((int64_t)key_cache.size() <= layer_idx) {
            return 0;
        }
        return current_length;
    }
    py::object get_max_length() {
        return py::none();
    }

    torch::Tensor update_key(torch::Tensor key_states, int64_t layer_idx) {
        int64_t next_length = current_length + key_states.size(-2);
        if (next_length <= max_seq_length) {
            key_cache[layer_idx].slice(2, current_length, next_length) = key_states;
        } else {
            key_cache[layer_idx] = std::move(torch::cat({key_cache[layer_idx], key_states}, -2));
        }
        return key_cache[layer_idx].slice(2, 0, next_length);
    }

    torch::Tensor update_value(torch::Tensor value_states, int64_t layer_idx) {
        int64_t next_length = current_length + value_states.size(-2);
        if (next_length <= max_seq_length) {
            value_cache[layer_idx].slice(2, current_length, next_length) = value_states;
        } else {
            value_cache[layer_idx] = std::move(torch::cat({value_cache[layer_idx], value_states}, -2));
        }
        return value_cache[layer_idx].slice(2, 0, next_length);
    }

    void length_update(int64_t x) {
        current_length = x;
    }

    int64_t get_usable_length(int new_seq_length, int layer_idx=0) {
        return get_seq_length(layer_idx);
    }

    std::vector<torch::Tensor> key_cache;
    std::vector<torch::Tensor> value_cache;
    int64_t current_length;
    int64_t max_seq_length;
};

struct DecoderLayer {
    DecoderLayer(py::object config, int64_t layer) {
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
        rms_norm_eps = config.attr("rms_norm_eps").cast<double>();
        is_causal = true;
        resid_pdrop = config.attr("resid_pdrop").cast<double>();

        // init rope
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        inv_freq = 1.0 / (torch::pow(config.attr("rope_theta").cast<double>(), torch::arange(0, head_dim, 2, options) / head_dim));
        div = std::sqrt(static_cast<float>(head_dim));
        pos = num_heads * head_dim;
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& attention_mask, const torch::Tensor& position_ids, Cache& past_key_value, bool output_attentions) {
        torch::NoGradGuard no_grad;

        // input_layernorm
        auto hidden_states = input_layernorm * RMSnorm_forward(x, rms_norm_eps);

        // qkv_split
        auto qkv = torch::matmul(hidden_states, qkv_proj.t());
        auto bsz = qkv.size(0);
        auto q_len = qkv.size(1);
        auto query_states = qkv.slice(-1, 0, pos);
        auto key_states = qkv.slice(-1, pos, pos + num_key_value_heads * head_dim);
        auto value_states = qkv.slice(-1, pos + num_key_value_heads * head_dim, qkv.size(-1));

        query_states = query_states.view({bsz, q_len, num_heads, head_dim}).transpose(1, 2);
        key_states = key_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);
        value_states = value_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);

        // rotary_embed
        auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(2).expand({position_ids.size(0), -1, 1});
        auto position_ids_expanded = position_ids.unsqueeze(1);
        // Force float32 since bfloat16 loses precision on long contexts
        // FIXME: implement torch.autocast()
        auto freqs = torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
        auto emb = torch::cat({freqs, freqs}, -1).to(value_states.dtype());
        auto cos = emb.cos();
        auto sin = emb.sin();

        // apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim = 1);
        // FIXME: handling position_ids=None and unsqueeze_dim = 1
        cos = cos.unsqueeze(1); //unsqueeze_dim
        sin = sin.unsqueeze(1);
        query_states = (query_states * cos) + (rotate_half(query_states) * sin);
        key_states = (key_states * cos) + (rotate_half(key_states) * sin);

        // attnetion forward & cache update
        // past_key_value.update(key_states, value_states, layer_idx);
        key_states = repeat_kv(past_key_value.update_key(key_states, layer_idx), num_key_value_groups);

        // reuse tensor, attn_weight -> query_states
        query_states = torch::matmul(query_states, key_states.transpose(2, 3)) / div;
        query_states = query_states + attention_mask;
        query_states = torch::nn::functional::softmax(query_states, torch::nn::functional::SoftmaxFuncOptions(-1.f)).to(value_states.scalar_type());

        // update value-cache
        value_states = repeat_kv(past_key_value.update_value(value_states, layer_idx), num_key_value_groups);
        if (layer_idx == 39) past_key_value.length_update(key_states.size(2));        
        // reuse tensor, attn_output -> value_states
        value_states = torch::matmul(query_states, value_states);
        value_states = value_states.transpose(1, 2).contiguous();
        value_states = value_states.reshape({bsz, q_len, hidden_size});
        value_states = torch::matmul(value_states, o_proj.t());

        // post_attention_layernorm
        // reuse tensor, residual -> key_states
        key_states = x + value_states;
        // reuse
        hidden_states = post_attention_layernorm * RMSnorm_forward(key_states, rms_norm_eps);

        // mlp
        // gate_up_proj
        auto up_states = torch::matmul(hidden_states, gate_up_proj.t());
        std::vector<torch::Tensor> chunks = up_states.chunk(2, -1);

        // down_proj
        return key_states + torch::matmul(chunks[1] * torch::silu(chunks[0]), down_proj.t());
    }

    bool set_weight (torch::Tensor qkv, torch::Tensor o, torch::Tensor input_norm, torch::Tensor post_norm, torch::Tensor up, torch::Tensor down) {
        qkv_proj = qkv;
        o_proj = o;
        input_layernorm = input_norm;
        post_attention_layernorm = post_norm;
        gate_up_proj = up;
        down_proj = down;

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
    double rms_norm_eps;
    double resid_pdrop;
    bool is_causal;
    float div;
    int pos;
    torch::Tensor qkv_proj;
    torch::Tensor o_proj;
    torch::Tensor inv_freq;
    torch::Tensor input_layernorm;
    torch::Tensor post_attention_layernorm;
    torch::Tensor gate_up_proj;
    torch::Tensor down_proj;
    // struct Cache_skkuter cache;
};

struct lm_head {
    torch::Tensor lm_head;
    void set_weight(torch::Tensor x) {
        lm_head = x;
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::NoGradGuard no_grad;
        return torch::matmul(x, lm_head.t()).to(torch::kFloat);
    }
};

struct Model {
    Model(py::object config) {
        padding_idx = config.attr("pad_token_id").cast<int64_t>();
        vocab_size = config.attr("vocab_size").cast<int64_t>();
        num_hidden_layers = config.attr("num_hidden_layers").cast<int64_t>();
        eps = config.attr("rms_norm_eps").cast<double>();
        sliding_window = config.attr("sliding_window").cast<int>();

        for (int i = 0; i < num_hidden_layers; i++) {
            layers.emplace_back(config, i);
        }
    }

    bool weight_copy(torch::Tensor embed_, torch::Tensor norm_) {
        embed = embed_;
        norm = norm_;
        return true;
    }

    bool decoder_weight_copy(const py::tuple& t, int64_t layer_idx) {
        // set_weight()
        if (t.size() != 6) {
            throw std::runtime_error("Expected 6 tensors");
            return false;
        }

        layers[layer_idx].set_weight(t[0].cast<torch::Tensor>(), t[1].cast<torch::Tensor>(), t[2].cast<torch::Tensor>(), t[3].cast<torch::Tensor>(), t[4].cast<torch::Tensor>(), t[5].cast<torch::Tensor>());
        return true;
    }

    std::tuple<torch::Tensor, py::tuple> forward(
        const torch::Tensor& input_ids,
        const torch::optional<torch::Tensor>& input_embeds,
        torch::Tensor attention_mask,
        torch::optional<torch::Tensor> position_ids_,
        Cache& past_key_values,
        bool output_attentions,
        bool output_hidden_states,
        bool use_cache) {
        torch::NoGradGuard no_grad;

        torch::Tensor x;
        torch::Tensor position_ids;
        int past_key_values_length;

        // retrieve input_ids and inputs_embeds
        // TODO:

        // embedding forward
        if (input_embeds.has_value()) {
            x = input_embeds.value();
        } else { // if input_embeds is None:
            x = torch::embedding(embed, input_ids);
        }

        auto seq_length = x.size(1);
        auto device = x.device();
        // Cache handling
        if (use_cache) {
            past_key_values_length = past_key_values.get_usable_length(seq_length);
        }

        // position_ids
        if (position_ids_.has_value()) {
            position_ids = position_ids_.value();
            position_ids = position_ids.view({-1, seq_length}).to(torch::kFloat);
        } else { // if position_ids_ is None:
            if (input_ids.defined()) device = input_ids.device();
            position_ids = torch::arange(past_key_values_length, seq_length + past_key_values_length,
                                                torch::TensorOptions().dtype(torch::kFloat).device(device));
            position_ids = position_ids.unsqueeze(0).view({-1, seq_length});
        }

        // prepare attn_mask
        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                x,
                past_key_values_length,
                sliding_window
            );

        // forward pass of model
        auto all_hidden_states = py::make_tuple();
        for (auto& layer : layers) {
            if (output_hidden_states) {
                all_hidden_states += py::make_tuple(x);
            }
            x = layer.forward(
                x,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions);
        }

        // Normalization
        x = norm * RMSnorm_forward(x, eps);

        // add hidden states from the last decoder layer
        if (output_hidden_states) all_hidden_states += py::make_tuple(x);

        return std::make_tuple(x, all_hidden_states);
        // FIXME: 구현 필요
        // if output_attentions: all_self_attns += (layer_outputs[1],)
    }

    int64_t padding_idx;
    int64_t vocab_size;
    int64_t num_hidden_layers;
    double eps;
    int64_t sliding_window;

    std::vector<DecoderLayer> layers;
    torch::Tensor norm;
    torch::Tensor embed;
};

float finfo(torch::Dtype dtype) {
    if (dtype == torch::kFloat32) {
        return std::numeric_limits<float>::lowest();
    } else if (dtype == torch::kFloat64) {
        return std::numeric_limits<double>::lowest();
    } else if (dtype == torch::kFloat16) {
        return -65504.0f; // Half-precision floating point
    } else if (dtype == torch::kBFloat16) {
        return -3.38953139e+38f; // BFloat16
    } else if (dtype == torch::kInt32 || dtype == torch::kInt64 || dtype == torch::kInt16) {
        return std::numeric_limits<int>::lowest();
    } else {
        // For unsupported types, return the most negative possible float
        return std::numeric_limits<float>::lowest();
    }
}

// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py#L298
// Convert `_prepare_4d_causal_attention_mask` by torch c++ binding
// remove class `AttentionMaskConverter` and merge `_prepare_4d_causal_attention_mask`, `to_4d`, `_expand_mask` and `_make_causal_mask`
torch::Tensor _prepare_4d_causal_attention_mask(
    torch::Tensor attention_mask,
    torch::Tensor inputs_embeds,
    int64_t past_key_values_length,
    int64_t sliding_window = -1) {

    int64_t bsz = inputs_embeds.size(0);
    int64_t seq_length = inputs_embeds.size(1);

    auto key_value_length = seq_length + past_key_values_length;
    float min_value = finfo(inputs_embeds.scalar_type());
    torch::Dtype dtype = inputs_embeds.scalar_type();

    // 4d mask is passed through the layers
    if (attention_mask.defined() && attention_mask.dim() == 2) {
        // `to_4d`
        // (torch::Tensor attention_mask_2d, int64_t query_length, int key_value_length, torch::Dtype dtype, int sliding_window) -> torch::Tensor
        // create causal mask
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        torch::Tensor causal_4d_mask;
        if (seq_length > 1) { // is_causal is always true
            // `_make_causal_mask`
            // (int batch_size, int query_length, int key_value_length, torch::Dtype dtype, torch::Device device, int past_key_values_length, int sliding_window = -1) -> torch::Tensor
            // Make causal mask used for bi-directional self-attention.
            auto device = attention_mask.device();
            auto mask = torch::full({seq_length, seq_length}, min_value, torch::TensorOptions().dtype(dtype).device(device));
            auto mask_cond = torch::arange(seq_length, torch::TensorOptions().device(device));
            mask.masked_fill_(mask_cond < (mask_cond.unsqueeze(-1)+1), 0);
            // auto mask_cond = torch::arange(mask.size(-1), torch::TensorOptions().device(device));
            // mask.masked_fill_(mask_cond < (mask_cond + 1).reshape({attention_mask.sizes()[attention_mask.dim() - 1], 1}), 0);

            if (past_key_values_length > 0) {
                mask = torch::cat({torch::zeros({seq_length, past_key_values_length}, torch::TensorOptions().dtype(dtype).device(device)), mask}, -1);
            }

            // add lower triangular sliding window mask if necessary
            if (sliding_window > 0) {
                auto diagonal = past_key_values_length - sliding_window + 1;
                auto context_mask = 1 - torch::triu(torch::ones_like(mask), diagonal);
                mask.masked_fill_(context_mask.to(torch::kBool), min_value);
            }

            causal_4d_mask = mask.unsqueeze(0).unsqueeze(0).expand({bsz, 1, seq_length, key_value_length});
        }

        // `_expand_mask`
        // (torch::Tensor mask, torch::Dtype dtype, int64_t tgt_len = -1) -> torch::Tensor
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        auto expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand({bsz, 1, seq_length, attention_mask.size(1)}).to(inputs_embeds.scalar_type());
        auto inverted_mask = 1.0 - expanded_mask;
        auto expanded_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch::kBool), min_value);

        if (causal_4d_mask.defined()) {
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.to(torch::kBool), min_value);
        }

        attention_mask = expanded_attn_mask;

    } else if (attention_mask.defined() && attention_mask.dim() == 4) {
        std::vector<int64_t> expected_shape = {bsz, 1, seq_length, key_value_length};
        if (attention_mask.sizes() != torch::IntArrayRef(expected_shape)) {
            throw std::invalid_argument("Incorrect 4D attention_mask shape");
        } else {
            // if the 4D mask has correct shape - invert it and fill with negative infinity
            auto inverted_mask = 1.0 - attention_mask;
            attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch::kBool), min_value);
        }
    } else {
        // TODO: implement `to_causal_4d` (if needed)
    }

    return attention_mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_kv", &repeat_kv, "repeat_kv");
    m.def("RMSnorm_forward", &RMSnorm_forward, "RMSnorm_forward");
    m.def("_prepare_4d_causal_attention_mask", &_prepare_4d_causal_attention_mask, "_prepare_4d_causal_attention_mask");

    py::class_<DecoderLayer, std::shared_ptr<DecoderLayer>>(m, "DecoderLayer")
        .def(py::init<py::object, int64_t>())
        .def("__call__", &DecoderLayer::forward)
        .def("forward", &DecoderLayer::forward)
        .def("set_weight", &DecoderLayer::set_weight);
    py::class_<lm_head, std::shared_ptr<lm_head>>(m, "lm_head")
        .def(py::init<>())
        .def("__call__", &lm_head::forward)
        .def("set_weight", &lm_head::set_weight)
        .def("forward", &lm_head::forward);
    py::class_<Model, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<py::object>())
        .def("__call__", &Model::forward)
        .def("weight_copy", &Model::weight_copy)
        .def("decoder_weight_copy", &Model::decoder_weight_copy)
        .def("forward", &Model::forward);
    py::class_<Cache, std::shared_ptr<Cache>>(m, "Cache")
        .def(py::init<int64_t, int64_t, int64_t>())
        .def("get_seq_length", &Cache::get_seq_length)
        .def("get_max_length", &Cache::get_max_length)
        .def("get_usable_length", &Cache::get_usable_length)
        .def("update_key", &Cache::update_key)
        .def("update_value", &Cache::update_value)
        .def("length_update", &Cache::length_update);
}