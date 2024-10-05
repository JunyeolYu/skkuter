


// #include <torch/types.h>
// torch::Tensor cuda_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask);


void myTest();
torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask);
torch::Tensor post_attention_forward(torch::Tensor value_state, torch::Tensor o_proj, torch::Tensor x);
torch::Tensor rms_forward(torch::Tensor hidden_states,  double rms_norm_epsilon, torch::Tensor rms_norm_weight);
