# Selected experiments

These notes summarize the existing research archive, whose recorded status is 2026-10-07. Metric values are historical observations. The compact release does not contain the full experiment archive or large training assets.

## Language and vision foundations

- **MiniMind:** random-initialization pretraining followed by full SFT, using the official tokenizer and mini datasets. Width 768, 8 layers, sequence length 768, BF16. Pretraining: one epoch, LR 5e-4, accumulation 4; SFT: one epoch, LR 1e-5, accumulation 1; microbatch 16. Exact-record hashing held out 1%. The two recorded NLLs use different supervised evaluation sets and should not be treated as a paired improvement.
- **ViT:** width 192, depth 6, 3 heads; CIFAR-10, 45,000/5,000 train/validation images, 30 epochs. Patch-4 and patch-8 results use seeds 42/43/44. [Original summary](../results/vit.json)
- **CLIP:** frozen pretrained towers with two 512→512 linear adapters and a learned logit scale; 524,289 trainable parameters. Flickr8k uses image-byte deduplication, 5,999 training images and full-gallery retrieval. The adapters are not reused automatically in the captioner. [Original summary](../results/clip.json)

## Visual connection and compression

Frozen CLIP ViT-B/32 supplies 49 patch tokens, excluding CLS and the global image projection. The projector is LayerNorm→Linear→GELU→Linear. Frozen language weights still carry gradients back to the projector.

The main caption comparison trains the projector and the last language layer. All arms use the same SFT starting point and three seeds. On the fixed Flickr8k caption test, greedy decoding gives:

| Visual representation | Added compressor parameters | CIDEr, mean ± sample SD |
|---|---:|---:|
| Raw 49 patches | 0 | 0.3363 ± 0.0154 |
| Fixed 16-slot pooling | 0 | 0.3050 ± 0.0237 |
| Learned local spatial weights | 100 | 0.3027 ± 0.0210 |
| Content-dependent query residual | 200,208 | 0.3138 ± 0.0196 |

[Extracted source metrics](../results/compression.json). Query gates start at zero, preserving fixed-pool outputs initially. The query branch initially receives zero loss gradient; its gates can open that path during optimization. Parameter-matched content controls supported a directional content effect, while added 2D position encodings showed no stable benefit.

Fixed pooling reduced batch-32 language prefill from about 3.98 to 2.97 ms and prompt KV storage by 40.2% in the recorded benchmark. This is not an end-to-end generation speedup. Query advantages were not consistent after COCO retraining.

## Quality and factuality

A fixed 1.1 repetition penalty reduced repetition and increased CIDEr, but also increased CHAIR object-unsupported metrics. A later image-breadth experiment held updates and image presentations fixed while expanding training images from 5,999 to 17,997. On its new test subset, mean CIDEr rose from 0.4469 to 0.4644; CHAIR_s worsened from 44.37% to 45.20%.

Object-word alignment, replay, freezing and KL controls produced mixed tradeoffs and did not consistently beat the original caption starting point. That tuning sequence was closed. CHAIR is annotation-dependent object support, not a complete measure of scene truth.

## Visual question answering

QP updates the projector; QL1 also updates the last language layer. Both start from the caption model and train for 500 steps on 10,000 questions from 3,334 images.

| Model | Development accuracy | Reserved-subset accuracy |
|---|---:|---:|
| Original caption model, Q0 | 0.00% | 0.00% |
| Projector only, QP | 30.21% | 31.37% |
| Projector + last layer, QL1 | 30.67% | 30.73% |

The reserved subset contains 500 images and 1,500 questions. Model weights and decoding were locked before answers were scored. [Extracted reserved metrics](../results/vqa_reserved.json)

A question-prefix-only statistical baseline reached **30.15% on development**, close to the trained models on that same split. Question-conditioned gating and replacing repeated questions with new questions on the same images did not establish reliable gains on the primary image-paired comparisons; their intervals included zero. Caption retention also deteriorated.

## How to read the evidence

Three-seed summaries describe training variation. Image-bootstrap intervals describe conditional variation on the fixed observed images; they do not cover repeated model selection or training-seed uncertainty. Observed reserved images are historical data now and cannot serve as a fresh test set again.

[Source hashes](../results/provenance.json) identify the archived reports behind the compressed narrative. This project demonstrates an implementation and experimental workflow, with useful negative results, rather than a validated new method.
