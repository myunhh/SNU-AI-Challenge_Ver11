from snuai11.decompose import content_words, decompose_caption, split_clauses

REAL_CAPTIONS = [
    "The camera pans left to reveal two people kneeling beside a fishing hole "
    "on the ice, then shifts from a frontal to a rear view as a skier moves "
    "forward and jumps onto a rail.",
    "A girl hula hoops indoors before the scene shifts outdoors to a cheering "
    "group on rocks; then, players swim towards the pool's center, with one in "
    "a white cap preparing to pass the ball as spectators watch.",
    "The woman lowers her gaze to apply a yellow contact lens as the camera "
    "zooms in, followed by the child showcasing a small object, then moving to "
    "a seated position at a table as the camera zooms out.",
    "The man moves closer to the mirror, tilting his head up while shaving, "
    "then a towel is raised to a face as the camera zooms in on a hand "
    "reaching down to touch the water surface.",
]


def test_always_exactly_four_nonempty():
    for cap in REAL_CAPTIONS + ["Short.", "", "one two", "a, b, c, d, e, f, g"]:
        events = decompose_caption(cap)
        assert len(events) == 4
        assert all(isinstance(e, str) and e.strip() for e in events)


def test_deterministic():
    for cap in REAL_CAPTIONS:
        assert decompose_caption(cap) == decompose_caption(cap)


def test_splits_on_temporal_connectives():
    clauses = split_clauses(REAL_CAPTIONS[0])
    assert len(clauses) >= 2  # "then" must split
    joined = " ".join(clauses).lower()
    assert "fishing hole" in joined and "skier" in joined


def test_coverage_no_content_lost():
    # boundary connectives (then/before/...) are consumed as separators by
    # design — they carry ordering, not visual content.
    connectives = {"then", "before", "after", "afterwards", "next", "finally",
                   "subsequently", "meanwhile", "eventually", "later", "followed"}
    for cap in REAL_CAPTIONS:
        events = decompose_caption(cap)
        joined = " ".join(events).lower()
        for word in content_words(cap):
            if word in connectives:
                continue
            assert word in joined, f"lost {word!r} from {cap!r}"


def test_content_words_filters_stopwords():
    words = content_words("The man moves closer to the mirror")
    assert "the" not in words and "to" not in words
    assert "man" in words and "mirror" in words
    # never returns empty
    assert content_words("the of a") != []
