import torch

from snuai11 import perm
from snuai11.train_sft import LORA_SUFFIXES, margin_dpo_loss


def test_margin_dpo_loss_decreases_with_gt_margin():
    label = 5
    weak = torch.zeros(1, 24)
    strong = torch.zeros(1, 24)
    strong[0, label] = 5.0
    l_weak = margin_dpo_loss(weak, label, beta=1.0, ce_weight=0.0)
    l_strong = margin_dpo_loss(strong, label, beta=1.0, ce_weight=0.0)
    assert float(l_strong) < float(l_weak)


def test_margin_dpo_loss_targets_adjacent_swaps():
    label = 0  # rank (0,1,2,3)
    negs = [perm.index_of(n) for n in perm.adjacent_swap_neighbors(perm.rank_of_index(label))]
    # raising a NON-neighbor class must not change the pure margin loss
    logits = torch.zeros(1, 24)
    base = margin_dpo_loss(logits, label, beta=1.0, ce_weight=0.0)
    far = [c for c in range(24) if c not in negs and c != label][0]
    logits2 = logits.clone()
    logits2[0, far] = 3.0
    # (log_softmax shifts all classes, so allow tiny change; neighbor raise must hurt much more)
    l_far = margin_dpo_loss(logits2, label, beta=1.0, ce_weight=0.0)
    logits3 = logits.clone()
    logits3[0, negs[0]] = 3.0
    l_neg = margin_dpo_loss(logits3, label, beta=1.0, ce_weight=0.0)
    assert float(l_neg) > float(l_far) >= float(base) - 1e-6


def test_margin_dpo_loss_grad_flows():
    logits = torch.zeros(1, 24, requires_grad=True)
    loss = margin_dpo_loss(logits, 3, beta=1.0, ce_weight=0.2)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_lora_suffixes_language_only_conventions():
    assert "q_proj" in LORA_SUFFIXES and "down_proj" in LORA_SUFFIXES
    assert "lm_head" not in LORA_SUFFIXES and "embed_tokens" not in LORA_SUFFIXES
