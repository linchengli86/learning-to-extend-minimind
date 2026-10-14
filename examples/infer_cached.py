"""Greedy caption or question answering from one cached CLIP patch tensor."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from minimind_ext.captioner import Captioner, generate_ids
from minimind_ext.compression import LearnedCaptioner
from minimind_ext.data import make_example
from minimind_ext.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patches', type=Path, required=True)
    parser.add_argument('--image-index', type=int, default=0)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--llm-checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--question', default='Describe the image in one short English sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    mode = state['config']['compressor']
    count = 49 if mode == 'raw' else 16
    llm = MiniMindForCausalLM(MiniMindConfig())
    llm.load_state_dict(torch.load(args.llm_checkpoint, map_location='cpu', weights_only=True), strict=True)
    base = Captioner(llm, tokenizer.convert_tokens_to_ids('<|image_pad|>'), vision_tokens=count)
    base.projector.load_state_dict(state['projector'], strict=True)
    for index, weights in state['language_layers'].items():
        llm.model.layers[int(index)].load_state_dict(weights, strict=True)
    model = base if mode == 'raw' else LearnedCaptioner(base, mode)
    if mode != 'raw':
        model.compressor.load_state_dict(state['compressor'], strict=True)
    model.to(args.device).eval()
    patches = torch.load(args.patches, map_location='cpu', weights_only=True)[args.image_index]
    # An empty answer contributes only EOS; remove it to recover the user prompt.
    example = make_example(tokenizer, patches, args.question, '', count)
    prompt = torch.tensor([example['input_ids'][:-1]], device=args.device)
    generated = generate_ids(model, patches.unsqueeze(0).to(args.device), prompt,
                             tokenizer.eos_token_id, args.max_new_tokens)
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == '__main__':
    main()
