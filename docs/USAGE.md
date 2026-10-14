# Using the selected code

## Environment

The original runs used Python 3.12, PyTorch 2.8.0, Transformers 4.57.6 and CUDA 12.8 on an RTX 5090 D. CPU is sufficient for the small smoke example and tests. Install PyTorch for your hardware, then `python -m pip install -e .`.

The included tokenizer and language model source come from the pinned MiniMind revision. Full language pretraining/SFT runners remain in that upstream revision; the original local settings are recorded in [language_baseline.json](../configs/language_baseline.json).

## Local inputs

Training requires an existing compatible MiniMind SFT state dictionary and CLIP patch features. Neither is bundled. The generic example uses:

- `patches.pt`: a tensor `[N, 49, 768]`, loaded with `weights_only=True`.
- `train.jsonl`: one record per supervised answer, with `image_index`, `question`, and `answer`.
- `full_sft_768.pth`: the width-768, eight-layer MiniMind state dictionary.

Example JSONL record:

```json
{"image_index": 0, "question": "What color is the bus?", "answer": "yellow"}
```

For captioning, use `Describe the image in one short English sentence.` as the question. Create CLIP features with a local `CLIPVisionModel` and its matching processor:

```python
with torch.no_grad():
    patches = vision_model(**image_inputs).last_hidden_state[:, 1:, :]
assert patches.shape[1:] == (49, 768)
torch.save(patches.cpu(), "patches.pt")
```

Use CLIP ViT-B/32 at 224×224 with its original preprocessing. These are patch hidden states, not 512-dimensional global retrieval vectors. Store the model revision, image IDs and preprocessing identity alongside your own cache. Deduplicate and split images before constructing question or caption records.

## Train and infer

Run from the repository root:

```bash
python -m examples.train_cached \
  --records data/train.jsonl --patches data/patches.pt \
  --tokenizer tokenizer --llm-checkpoint checkpoints/full_sft_768.pth \
  --output checkpoints/captioner.pt --steps 500 --last-layers 0
```

Use `--compressor fixed`, `spatial` or `query` for 16-token input; the default `raw` retains 49. `--last-layers 1` enables the last decoder layer. Projector/compressor and language learning rates are configured separately. The checkpoint stores trained modules and configuration; the base language weights are still required.

```bash
python -m examples.infer_cached \
  --patches data/patches.pt --image-index 0 --tokenizer tokenizer \
  --llm-checkpoint checkpoints/full_sft_768.pth \
  --checkpoint checkpoints/captioner.pt --question "What color is the bus?"
```

These are small new examples with fixed-step training and greedy inference. They omit the historical runners' resume, validation-selection, identity-lock and complete evaluation machinery. Their outputs are new experiments and should not be reported as reproductions of the archived metrics.

## Score VQAv2 predictions

The scorer uses only the Python standard library. Predictions are a JSON list of `question_id`/`answer` records; official annotations supply ten reference answers per question. Question coverage is strict.

```bash
python -m minimind_ext.vqa_score \
  --annotations annotations.json --predictions predictions.json --output scores.json
```

The pinned official evaluator and its license are retained for comparison. Tests cover normalization/scoring agreement, cached decoding, frozen-language gradients and compressor initialization. Full GPU training and historical evaluations are separate from these checks.
