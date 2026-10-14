"""Visual placeholder replacement and greedy cached decoding.

Adapted for portable imports; historical run-path helpers have been removed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

class Captioner(nn.Module):
    def __init__(self, llm, image_token_id, vision_dim=768, vision_tokens=49):
        super().__init__()
        self.llm = llm.requires_grad_(False).eval()
        self.image_token_id, self.vision_tokens = image_token_id, vision_tokens
        hidden = llm.config.hidden_size
        self.projector = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden),
                                       nn.GELU(), nn.Linear(hidden, hidden))

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, input_ids, patches=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, logits_to_keep=0):
        backbone = self.llm.model
        batch, length = input_ids.shape
        past = past_key_values or [None] * len(backbone.layers)
        start = past[0][0].shape[1] if past[0] is not None else 0
        hidden = backbone.embed_tokens(input_ids)
        if patches is not None:
            if start:
                raise ValueError('patches belong to prefill only, not cached decoding')
            mask = input_ids.eq(self.image_token_id)
            if not torch.all(mask.sum(1) == self.vision_tokens):
                raise ValueError('each sample must contain exactly the configured image placeholders')
            if patches.shape[:2] != (batch, self.vision_tokens):
                raise ValueError('patch shape does not match placeholder count')
            projected = self.projector(patches.to(dtype=self.projector[1].weight.dtype))
            hidden = hidden.clone()
            hidden[mask] = projected.reshape(-1, hidden.shape[-1]).to(hidden.dtype)
        hidden = backbone.dropout(hidden)
        if start + length > len(backbone.freqs_cos):
            raise ValueError('sequence exceeds RoPE buffer')
        positions = (backbone.freqs_cos[start:start+length], backbone.freqs_sin[start:start+length])
        presents = []
        # Frozen weights still participate in autograd: input gradients are
        # essential for training the projector. Do not use no_grad here.
        for layer, previous in zip(backbone.layers, past):
            hidden, present = layer(hidden, positions, past_key_value=previous,
                                    use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        hidden = backbone.norm(hidden)
        if logits_to_keep and labels is not None:
            raise ValueError('full logits required for supervised loss')
        logits = self.llm.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        loss = None
        if labels is not None:
            if not (labels[:, 1:] != -100).any():
                raise ValueError('batch has no supervised answer tokens')
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=presents if use_cache else None)


@torch.inference_mode()
def generate_ids(model, patches, prompt_ids, eos_token_id, max_new_tokens=64, use_cache=True):
    model.eval()
    if prompt_ids.shape[0] != 1:
        raise ValueError('generation supports one image at a time')
    all_ids, past, generated = prompt_ids, None, []
    for _ in range(max_new_tokens):
        current = all_ids[:, -1:] if past is not None else all_ids
        output = model(current, patches=patches if past is None else None,
                       past_key_values=past, use_cache=use_cache, logits_to_keep=1)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token.item())
        all_ids = torch.cat((all_ids, token), dim=1)
        past = output.past_key_values if use_cache else None
        if token.item() == eos_token_id:
            break
    return generated
