#pragma once

#include <torch/extension.h>
#include <kerutils/supplemental/torch_tensors.h>

std::vector<at::Tensor> sparse_attn_prefill_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
);
