# Attribution and curation

This repository selects code and findings from the existing MiniMind experiment archive. Original source paths and SHA-256 hashes are recorded in [results/provenance.json](results/provenance.json).

| Material | Origin and terms |
|---|---|
| `minimind_ext/model_minimind.py`, `tokenizer/` | [jingyaogong/minimind](https://github.com/jingyaogong/minimind), commit `f659b55761b754d306bd140573493a6543cafd7f`. Kept byte-for-byte. Original Apache 2.0 license: [licenses/MINIMIND.txt](licenses/MINIMIND.txt). |
| VQA normalization tables and reference evaluator | [GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA), commit `a013f0043c1e2cdc995922dfe257f7149aa9af06`. Original and archived Python 3 reference files are retained in `third_party/vqa/`; original terms are in [licenses/VQA.txt](licenses/VQA.txt). |
| ViT, CLIP and visual-language experiments | Existing local project implementations. Portable copies select classes/functions and replace imports or private artifact paths where needed. |
| English notes, packaging and examples | Added during this curation. New examples are not the historical experiment runners. |

The project does not assign a new blanket license to the local experiment code. Third-party material retains its original terms. No image datasets, pretrained CLIP weights or generated dataset examples are redistributed.

The original MiniMind source includes optional paths such as MoE and extended RoPE. Their presence does not mean those paths were evaluated in these experiments. The captioner draws on the visual-placeholder design of MiniMind-V and LLaVA-style projection; this project does not claim a full reproduction of either system.
