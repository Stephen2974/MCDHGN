"""RankNet-inspired sampled pairwise logistic loss on training nodes only."""
import torch
import torch.nn.functional as F


def training_objective(logits, labels, class_weight, config, epoch):
    settings = config.get('ranking', {})
    coefficient = float(settings.get('coefficient', 0.0))
    if coefficient < 0:
        raise ValueError('Ranking coefficient must be nonnegative')
    ce = F.cross_entropy(logits, labels, weight=class_weight)
    if coefficient == 0:
        return ce  # Exact baseline operation; no RNG consumed.
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(labels):
        raise ValueError('Expected aligned binary training logits and labels')
    positive = torch.nonzero(labels == 1, as_tuple=False).reshape(-1)
    negative = torch.nonzero(labels == 0, as_tuple=False).reshape(-1)
    if not len(positive) or not len(negative):
        raise ValueError('Both training classes are required')
    count = int(settings.get('pairs', 8192))
    if count < 1:
        raise ValueError('Need at least one pair')
    # Separate CPU generator keeps dropout and model RNG streams unchanged.
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(config['seed']) + 1000 * int(config['fold']) + 1000003 * int(epoch))
    pi = torch.randint(len(positive), (count,), generator=gen).to(logits.device)
    ni = torch.randint(len(negative), (count,), generator=gen).to(logits.device)
    margin = logits[:, 1] - logits[:, 0]
    rank = F.softplus(margin[negative[ni]] - margin[positive[pi]]).mean()
    return ce + coefficient * rank
