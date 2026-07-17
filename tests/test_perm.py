import itertools

import pytest

from snuai11 import perm


def test_index_roundtrip():
    for c in range(24):
        assert perm.index_of(perm.rank_of_index(c)) == c


def test_letters():
    assert perm.LETTERS[0] == "A" and perm.LETTERS[23] == "X"
    for c in range(24):
        assert perm.index_of_letter(perm.letter_of_index(c)) == c


def test_invert_is_involution_and_inverse():
    for p in perm.ALL_PERMS:
        inv = perm.invert(p)
        assert perm.invert(inv) == p
        assert perm.compose(p, inv) == perm.IDENTITY
        assert perm.compose(inv, p) == perm.IDENTITY


def test_order_rank_bridge_non_self_inverse():
    # order (1,2,0,3): step0 shows input1, step1 input2, step2 input0.
    # rank must be: input0 -> step2, input1 -> step0, input2 -> step1.
    assert perm.order_to_rank((1, 2, 0, 3)) == (2, 0, 1, 3)
    # the famous trap: index_of(order) != index_of(rank) for non-self-inverse
    order = (1, 2, 0, 3)
    assert perm.index_of(order) != perm.index_of(perm.order_to_rank(order))
    # exactly half the permutations of S4 with fixed structure are self-inverse (here 10 of 24)
    self_inv = [p for p in perm.ALL_PERMS if perm.invert(p) == p]
    assert len(self_inv) == 10


def test_view_roundtrip_exhaustive():
    # shuffling images by sigma then remapping the view label must recover
    # the original label — for all 24x24 combinations.
    for rank in perm.ALL_PERMS:
        for sigma in perm.ALL_PERMS:
            rv = perm.apply_view(rank, sigma)
            assert perm.unapply_view(rv, sigma) == rank


def test_view_class_map_bijective():
    for sigma in perm.ALL_PERMS:
        m = perm.view_class_map(sigma)
        assert sorted(m) == list(range(24))
    assert perm.view_class_map(perm.IDENTITY) == list(range(24))


def test_kendall_and_pairwise():
    assert perm.kendall_distance((0, 1, 2, 3), (0, 1, 2, 3)) == 0
    assert perm.kendall_distance((0, 1, 2, 3), (3, 2, 1, 0)) == 6
    assert perm.pairwise_score((0, 1, 2, 3), (3, 2, 1, 0)) == 0.0
    assert perm.pairwise_score((0, 1, 2, 3), (0, 1, 3, 2)) == pytest.approx(5 / 6)


def test_adjacent_swap_neighbors():
    for rank in perm.ALL_PERMS:
        negs = perm.adjacent_swap_neighbors(rank)
        assert len(negs) == 3
        assert len(set(negs)) == 3
        for n in negs:
            assert perm.kendall_distance(rank, n) == 1
            assert n != rank
    # exhaustive: KT-1 neighbors are exactly the adjacent transpositions
    for rank in perm.ALL_PERMS:
        all_kt1 = [q for q in perm.ALL_PERMS if perm.kendall_distance(rank, q) == 1]
        assert sorted(all_kt1) == sorted(perm.adjacent_swap_neighbors(rank))


def test_all_perms_is_lexicographic():
    assert perm.ALL_PERMS == list(itertools.permutations(range(4)))


def test_kendall_matrix_matches_pairwise_distances():
    m = perm.kendall_matrix()
    assert len(m) == 24 and all(len(row) == 24 for row in m)
    for a in range(24):
        assert m[a][a] == 0
        for b in range(24):
            assert m[a][b] == m[b][a]
            assert m[a][b] == perm.kendall_distance(perm.ALL_PERMS[a], perm.ALL_PERMS[b])
    # S4 distance distribution from any fixed rank: 1/3/5/6/5/3/1 -> mean 3
    for a in range(24):
        assert sum(m[a]) == 72
