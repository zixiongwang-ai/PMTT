"""SPair PCK at 0.1 of the target bounding-box extent."""
import torch

def pck(scores, xy, size, box):
    h = w = 64
    index = scores.argmax(-1)
    pred = torch.stack([(index % w + 0.5) / w * size[0], (index // w + 0.5) / h * size[1]], -1)
    ok = (pred - xy).norm(dim=-1) <= 0.1 * max(box[2] - box[0], box[3] - box[1])
    correct = (xy[:, 1] / size[1] * h).long().clamp(0, h - 1) * w + (xy[:, 0] / size[0] * w).long().clamp(0, w - 1)
    true = scores.gather(1, correct[:, None]).squeeze(1)
    rank = (scores > true[:, None]).sum(-1) + 1
    wrong = scores.clone()
    wrong.scatter_(1, correct[:, None], -torch.inf)
    return dict(ok=ok.cpu(), rank=rank.cpu(), correct_score=true.cpu(), wrong_score=wrong.max(-1).values.cpu())
