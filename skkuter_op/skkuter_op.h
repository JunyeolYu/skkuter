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

torch::Tensor RMSnorm_forward(torch::Tensor hidden_states, double eps);

torch::Tensor _prepare_4d_causal_attention_mask(
    torch::Tensor attention_mask,
    torch::Tensor inputs_embeds,
    int64_t past_key_values_length,
    int64_t sliding_window);

torch::Tensor rotate_half(torch::Tensor x);

#endif // SKKUTER_OP_H