"""Assistant-only supervision shared by captioning and short-answer VQA."""
import torch


def make_example(tokenizer, patches, question, answer, vision_tokens=49, max_length=192):
    if vision_tokens not in (16, 49) or '<|' in question or '<|' in answer:
        raise ValueError('invalid visual token count or special-token markup')
    for token in ('<|vision_start|>', '<|image_pad|>', '<|vision_end|>'):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id or tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'{token} must already exist as one tokenizer token')
    image = '<|vision_start|>' + '<|image_pad|>' * vision_tokens + '<|vision_end|>'
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': image + '\n' + question}],
        tokenize=True, add_generation_prompt=True, open_thinking=False,
    )
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > max_length:
        raise ValueError('example exceeds max_length; truncation would change supervision')
    if patches.shape != (49, 768):
        raise ValueError('provide raw [49, 768] CLIP patches before optional compression')
    return {'input_ids': prompt + target, 'labels': [-100] * len(prompt) + target,
            'patches': patches}


def collate_examples(items, pad_token_id):
    length = max(len(item['input_ids']) for item in items)
    inputs = torch.full((len(items), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for row, item in enumerate(items):
        size = len(item['input_ids'])
        inputs[row, :size] = torch.tensor(item['input_ids'])
        labels[row, :size] = torch.tensor(item['labels'])
        mask[row, :size] = 1
    return {'input_ids': inputs, 'labels': labels, 'attention_mask': mask,
            'patches': torch.stack([item['patches'] for item in items])}
