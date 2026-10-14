"""Check behavior affected by extracting the portable core."""
import unittest
import torch
from transformers import AutoTokenizer
from pathlib import Path

from minimind_ext.captioner import Captioner
from minimind_ext.clip import contrastive_loss, retrieval_metrics
from minimind_ext.compression import LearnedCaptioner, QueryPool, SpatialPool, pool_patches
from minimind_ext.data import make_example, collate_examples
from minimind_ext.model_minimind import MiniMindConfig, MiniMindForCausalLM
from minimind_ext.vit import SmallViT


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def captioner(self):
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, vocab_size=128,
                               num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=128, flash_attn=False)
        return Captioner(MiniMindForCausalLM(config), image_token_id=10, vision_tokens=16).eval()

    def test_projector_gradient_crosses_frozen_backbone(self):
        model = self.captioner()
        ids = torch.tensor([[1] + [10] * 16 + [20, 21, 2]])
        labels = torch.full_like(ids, -100)
        labels[:, -2:] = ids[:, -2:]
        model(ids, patches=torch.randn(1, 16, 768), labels=labels).loss.backward()
        self.assertGreater(model.projector[1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.llm.parameters()))

    def test_cached_logits_match_full_prefix(self):
        model = self.captioner()
        ids = torch.tensor([[1] + [10] * 16 + [20]])
        patches = torch.randn(1, 16, 768)
        with torch.no_grad():
            first = model(ids, patches=patches, use_cache=True)
            next_id = torch.tensor([[21]])
            cached = model(next_id, past_key_values=first.past_key_values, use_cache=True)
            full = model(torch.cat([ids, next_id], dim=1), patches=patches)
        torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)

    def test_compressed_wrapper_prefill_and_cached_decode(self):
        model = LearnedCaptioner(self.captioner(), 'query').eval()
        ids = torch.tensor([[1] + [10] * 16 + [20]])
        with torch.no_grad():
            first = model(ids, patches=torch.randn(1, 49, 768), use_cache=True)
            next_output = model(torch.tensor([[21]]), past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(tuple(next_output.logits.shape), (1, 1, 128))

    def test_compressors_preserve_pooling_at_initialization(self):
        patches = torch.randn(2, 49, 768)
        expected = pool_patches(patches, 16)
        for module, parameters in ((SpatialPool(), 100), (QueryPool(), 200208)):
            self.assertEqual(sum(p.numel() for p in module.parameters()), parameters)
            torch.testing.assert_close(module(patches), expected, rtol=0, atol=0)

    def test_query_gates_open_the_attention_gradient(self):
        module = QueryPool()
        patches = torch.randn(2, 49, 768)
        module(patches).square().mean().backward()
        self.assertEqual(module.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.gates.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.gates.add_(-0.1 * module.gates.grad)
        module.zero_grad(set_to_none=True)
        module(patches).square().mean().backward()
        self.assertGreater(module.q_proj.weight.grad.abs().sum().item(), 0)

    def test_multi_positive_loss_does_not_create_false_negatives(self):
        features = torch.eye(3)
        scale = torch.tensor(1.0)
        ordinary = contrastive_loss(features, features, scale)
        ids = torch.arange(3)
        matched = contrastive_loss(features, features, scale, ids, ids)
        torch.testing.assert_close(ordinary, matched)
        all_same = torch.zeros(3, dtype=torch.long)
        self.assertAlmostEqual(contrastive_loss(features, features, scale, all_same, all_same).item(), 0)
        self.assertEqual(retrieval_metrics(features, features, ids, ids)['image_to_text']['R@1'], 1)

    def test_tokenizer_and_assistant_mask(self):
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / 'tokenizer',
                                                 local_files_only=True)
        item = make_example(tokenizer, torch.randn(49, 768), 'What color?', 'yellow', 16)
        image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.assertEqual(item['input_ids'].count(image_id), 16)
        self.assertEqual(item['labels'][-1], tokenizer.eos_token_id)
        self.assertEqual(sum(value == -100 for value in item['labels']),
                         len(item['labels']) - len(tokenizer.encode('yellow', add_special_tokens=False)) - 1)
        batch = collate_examples([item], tokenizer.pad_token_id)
        self.assertTrue(torch.all(batch['attention_mask'] == 1))

    def test_vit_forward_and_backward(self):
        model = SmallViT(dim=48, depth=1, heads=3)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        logits.sum().backward()
        self.assertGreater(model.patch_embedding.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
