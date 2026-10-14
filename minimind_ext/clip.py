"""Symmetric CLIP loss, including multiple captions for the same image."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def contrastive_loss(images, texts, logit_scale, image_ids=None, text_image_ids=None):
    images, texts = F.normalize(images.float(), dim=-1), F.normalize(texts.float(), dim=-1)
    logits = logit_scale.clamp(max=math.log(100)).exp() * images @ texts.T
    if image_ids is None and text_image_ids is None:
        if len(images) != len(texts):
            raise ValueError('one-to-one loss requires equal image/text batch sizes')
        targets = torch.arange(len(images), device=images.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    if image_ids is None or text_image_ids is None:
        raise ValueError('provide both image and text image IDs')
    positive = image_ids[:, None] == text_image_ids[None, :]
    if not positive.any(1).all() or not positive.any(0).all():
        raise ValueError('every image and text anchor needs a positive match')
    # Sum probability over all matching captions; do not label other captions
    # of the same image as negatives. Reduces to ordinary CE for one positive.
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    image_loss = logits.logsumexp(1) - positive_logits.logsumexp(1)
    text_loss = logits.logsumexp(0) - positive_logits.logsumexp(0)
    return (image_loss.mean() + text_loss.mean()) / 2


@torch.inference_mode()
def retrieval_metrics(images, texts, image_ids, text_image_ids):
    scores = F.normalize(images.float(), dim=-1) @ F.normalize(texts.float(), dim=-1).T
    result = {}
    for name, matrix, query_ids, gallery_ids in [
        ('image_to_text', scores, image_ids, text_image_ids),
        ('text_to_image', scores.T, text_image_ids, image_ids),
    ]:
        order = matrix.topk(min(10, matrix.shape[1]), dim=1).indices
        matches = gallery_ids[order] == query_ids[:, None]
        result[name] = {f'R@{k}': matches[:, :min(k, matrix.shape[1])].any(1).float().mean().item()
                        for k in (1, 5, 10)}
    return result


class CLIPAdapters(nn.Module):
    def __init__(self, dim, logit_scale):
        super().__init__()
        self.image = nn.Linear(dim, dim, bias=False)
        self.text = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.image.weight.copy_(torch.eye(dim))
            self.text.weight.copy_(torch.eye(dim))
        self.logit_scale = nn.Parameter(logit_scale.clone())
