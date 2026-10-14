"""Exercise the core on random CPU tensors; no downloads or model weights."""
import torch
from minimind_ext.model_minimind import MiniMindConfig, MiniMindForCausalLM
from minimind_ext.captioner import Captioner
from minimind_ext.compression import QueryPool
from minimind_ext.vit import SmallViT


def main():
    torch.manual_seed(42)
    config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, vocab_size=128,
                           num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=128, flash_attn=False)
    model = Captioner(MiniMindForCausalLM(config), image_token_id=10, vision_tokens=16)
    patches = QueryPool()(torch.randn(1, 49, 768))
    ids = torch.tensor([[1] + [10] * 16 + [20, 21, 2]])
    labels = torch.full_like(ids, -100)
    labels[:, -2:] = ids[:, -2:]
    loss = model(ids, patches=patches, labels=labels).loss
    loss.backward()
    assert model.projector[1].weight.grad is not None
    assert all(p.grad is None for p in model.llm.parameters())
    logits = SmallViT(dim=48, depth=1, heads=3)(torch.randn(2, 3, 32, 32))
    print(f'Visual-language loss: {loss.item():.4f}; ViT output: {tuple(logits.shape)}')
    print('Projector gradients pass through the frozen language model.')


if __name__ == '__main__':
    main()
