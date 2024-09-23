#ifndef SKKUTER_OP_H
#define SKKUTER_OP_H

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "ATen/ATen.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <tuple>

// Function
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
    torch::Tensor o_proj);

torch::Tensor RMSnorm_forward(torch::Tensor hidden_states, double eps);

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::optional<torch::Tensor> position_id,
    int64_t unsqueeze_dim);

torch::Tensor _prepare_4d_causal_attention_mask(
    torch::Tensor attention_mask,
    int64_t bsz,
    int64_t seq_length,
    torch::Tensor inputs_embeds,
    int64_t past_key_values_length,
    int64_t sliding_window);

torch::Tensor rotate_half(torch::Tensor x);

#endif // SKKUTER_OP_H