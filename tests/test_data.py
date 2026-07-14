import random
from pathlib import Path

from snuai11 import perm
from snuai11.data import Sample, holdout_ids, shuffle_sample, split_train_holdout, uniform_augment


def _mk(sid: str, rank) -> Sample:
    paths = tuple(Path(f"/img/{sid}_{k}.jpg") for k in range(4))
    return Sample(id=sid, image_paths=paths, caption="cap", rank=rank, label=perm.index_of(rank))


def test_shuffle_identity_noop():
    s = _mk("x", (2, 0, 1, 3))
    out = shuffle_sample(s, perm.IDENTITY)
    assert out.image_paths == s.image_paths and out.rank == s.rank


def test_shuffle_label_consistency():
    # after shuffling, the image at new slot j must keep its chronological rank
    s = _mk("x", (2, 0, 1, 3))
    for sigma in perm.ALL_PERMS:
        out = shuffle_sample(s, sigma)
        for j in range(4):
            orig_slot = s.image_paths.index(out.image_paths[j])
            assert out.rank[j] == s.rank[orig_slot]
        assert out.label == perm.index_of(out.rank)


def test_shuffle_then_unapply_view_recovers():
    s = _mk("x", (3, 1, 0, 2))
    for sigma in perm.ALL_PERMS:
        out = shuffle_sample(s, sigma)
        assert perm.unapply_view(out.rank, sigma) == s.rank


def test_uniform_augment_covers_label_space():
    s = _mk("x", (0, 1, 2, 3))
    rng = random.Random(0)
    labels = {uniform_augment(s, rng).label for _ in range(2000)}
    assert len(labels) == 24


def test_holdout_deterministic_and_disjoint():
    samples = [_mk(f"id{i:05d}", (0, 1, 2, 3)) for i in range(2000)]
    h1 = holdout_ids(samples, 200)
    h2 = holdout_ids(list(reversed(samples)), 200)
    assert h1 == h2  # order-independent
    train, hold = split_train_holdout(samples, 200)
    assert len(hold) == 200 and len(train) == 1800
    assert not ({s.id for s in train} & {s.id for s in hold})
