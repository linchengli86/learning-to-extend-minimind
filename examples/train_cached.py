"""Minimal caption/VQA training on local CLIP patch caches.

This portable example is not the original historical experiment runner.
"""
import argparse
from functools import partial
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from minimind_ext.captioner import Captioner
from minimind_ext.compression import LearnedCaptioner
from minimind_ext.data import make_example, collate_examples
from minimind_ext.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--patches', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--llm-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compressor', choices=('raw', 'fixed', 'spatial', 'query'), default='raw')
    parser.add_argument('--last-layers', type=int, default=0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--llm-lr', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 0 <= args.last_layers <= 8:
        parser.error('steps/batch-size must be positive; last-layers must be 0..8')
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    patches = torch.load(args.patches, map_location='cpu', weights_only=True)
    records = [json.loads(line) for line in args.records.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        parser.error('records must not be empty')
    count = 49 if args.compressor == 'raw' else 16
    items = [make_example(tokenizer, patches[row['image_index']], row['question'], row['answer'], count)
             for row in records]
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True,
                        collate_fn=partial(collate_examples, pad_token_id=tokenizer.pad_token_id))
    llm = MiniMindForCausalLM(MiniMindConfig())
    llm.load_state_dict(torch.load(args.llm_checkpoint, map_location='cpu', weights_only=True), strict=True)
    base = Captioner(llm, tokenizer.convert_tokens_to_ids('<|image_pad|>'), vision_tokens=count)
    selected = list(range(8 - args.last_layers, 8))
    for index in selected:
        base.llm.model.layers[index].requires_grad_(True)
    model = base if args.compressor == 'raw' else LearnedCaptioner(base, args.compressor)
    model.to(args.device).train()
    groups = [{'params': base.projector.parameters(), 'lr': args.lr}]
    if args.compressor != 'raw':
        compressor_parameters = list(model.compressor.parameters())
        if compressor_parameters:
            groups.append({'params': compressor_parameters, 'lr': args.lr})
    if selected:
        groups.append({'params': [p for layer in selected for p in llm.model.layers[layer].parameters()],
                       'lr': args.llm_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=torch.device(args.device).type == 'cuda'):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f'step={step + 1} loss={loss.item():.4f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'projector': base.projector.state_dict(),
                'language_layers': {str(i): llm.model.layers[i].state_dict() for i in selected},
                'compressor': None if args.compressor == 'raw' else model.compressor.state_dict(),
                'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}},
               args.output)


if __name__ == '__main__':
    main()
