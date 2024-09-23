#include "skkuter_op.h"

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
    torch::NoGradGuard no_grad;
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
        // cache = Cache_skkuter();
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
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor attention_mask, torch::Tensor position_ids, py::object past_key_value, bool output_attentions) {
        torch::NoGradGuard no_grad;

        // input_layernorm
        auto hidden_states = input_layernorm * RMSnorm_forward(x, rms_norm_eps);
        
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
        py::tuple kv_res = past_key_value.attr("update")(key_states, value_states, layer_idx, cache_kwargs);

        // attnetion forward
        // repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(kv_res[0].cast<torch::Tensor>(), num_key_value_groups);
        value_states = repeat_kv(kv_res[1].cast<torch::Tensor>(), num_key_value_groups);

        auto attn_weights = torch::matmul(query_states, key_states.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));
        attn_weights = attn_weights + attention_mask;
        attn_weights = torch::nn::functional::softmax(attn_weights, torch::nn::functional::SoftmaxFuncOptions(-1).dtype(torch::kFloat32)).to(value_states.scalar_type());
        attn_weights = torch::nn::functional::dropout(attn_weights, torch::nn::functional::DropoutFuncOptions().p(attention_dropout));

        auto attn_output = torch::matmul(attn_weights, value_states);
        attn_output = attn_output.transpose(1, 2);//.contiguous();
        attn_output = attn_output.reshape({bsz, q_len, hidden_size});
        attn_output = torch::matmul(attn_output, o_proj.t());

        // post_attention_layernorm
        auto residual = x + torch::nn::functional::dropout(attn_output, torch::nn::functional::DropoutFuncOptions().p(resid_pdrop));

        // post_attention_layernorm
        hidden_states = post_attention_layernorm * RMSnorm_forward(residual, rms_norm_eps);

        // mlp
        // gate_up_proj
        auto up_states = torch::matmul(hidden_states, gate_up_proj.t());
        std::vector<torch::Tensor> chunks = up_states.chunk(2, -1);

        // down_proj
        hidden_states = torch::matmul(chunks[1] * torch::silu(chunks[0]), down_proj.t());

        // resid_mlp_dropout
        auto output = residual + torch::nn::functional::dropout(hidden_states, torch::nn::functional::DropoutFuncOptions().p(resid_pdrop));

        return output;
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

    torch::Tensor qkv_proj;
    torch::Tensor o_proj;
    torch::Tensor inv_freq;
    torch::Tensor input_layernorm;
    torch::Tensor post_attention_layernorm;
    torch::Tensor gate_up_proj;
    torch::Tensor down_proj;
    // struct Cache_skkuter cache;
};

struct Embedding {
    torch::Tensor emb;
    // store weight
    void set_weight(torch::Tensor x) {
        emb = x;
    }
    torch::Tensor forward(torch::Tensor x) {
        torch::NoGradGuard no_grad;
        return torch::embedding(emb, x);
    }
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
        torch::Tensor input_ids,
        torch::optional<torch::Tensor> input_embeds,
        torch::Tensor attention_mask,
        torch::optional<torch::Tensor> position_ids_,
        py::object past_key_values,
        bool output_attentions,
        bool output_hidden_states,
        int sliding_window,
        bool use_cache) {
        
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
            // TODO: Handle the case where legacy_cache is used
            // // ~ PYTHON ~
            // bool use_legacy_cache = !(dynamic_cast<Cache*>(past_key_values.get()));
            // if (use_legacy_cache) past_key_values = DynamicCache::from_legacy_cache(past_key_values);

            // Assume that we always use the Dynamic cache
            past_key_values_length = py::cast<int>(past_key_values.attr("get_usable_length")(seq_length));
        }

        // position_ids
        if (position_ids_.has_value()) { 
            position_ids = position_ids_.value();
            position_ids = position_ids.view({-1, seq_length}).to(torch::kInt64);
        } else { // if position_ids_ is None:
            if (input_ids.defined()) device = input_ids.device();
            position_ids = torch::arange(past_key_values_length, seq_length + past_key_values_length,
                                                torch::TensorOptions().dtype(torch::kInt64).device(device));
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
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "apply_rotary_pos_emb");
    m.def("attention_forward", &attention_forward, "Attention forward pass in C++");
    m.def("RMSnorm_forward", &RMSnorm_forward, "RMSnorm_forward");
    m.def("qkv_split", &qkv_split, "qkv_split");
    m.def("_prepare_4d_causal_attention_mask", &_prepare_4d_causal_attention_mask, "_prepare_4d_causal_attention_mask");
    py::class_<Phi3RotaryEmbedding>(m, "Phi3RotaryEmbedding")
        .def(py::init<int64_t, int64_t, double>())
        .def("forward", &Phi3RotaryEmbedding::forward);
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
    py::class_<Embedding, std::shared_ptr<Embedding>>(m, "Embedding")
        .def(py::init<>())
        .def("__call__", &Embedding::forward)
        .def("set_weight", &Embedding::set_weight)
        .def("forward", &Embedding::forward);
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
}