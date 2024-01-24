# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
    PyTorch Mixtral model MixtralSparseMoeBlock patch for Deepspeed Zero3.
    Tested for transformers==4.36.2.
"""

import torch
import torch.nn.functional as F

def mlp_forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


## Ref. https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
    topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    topk_weight = topk_weight.to(hidden_states.dtype)

    hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
    y = torch.empty_like(hidden_states)
    flat_topk_idx = topk_idx.view(-1)
    for i in range(self.num_experts):
        expert = self.experts[i]
        y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def replace_mixtral_moe_with_dense_impl():
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP

    MixtralBLockSparseTop2MLP.forward = mlp_forward
    MixtralSparseMoeBlock.forward = moe_forward
