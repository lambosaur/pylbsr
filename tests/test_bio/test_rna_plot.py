import random

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
pytest.importorskip("RNA")

import matplotlib.pyplot as plt
import RNA

from pylbsr.bio import rna_plot as rp

TRNA_PHE = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"


def _random_fold(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return RNA.fold("".join(rng.choice("ACGU") for _ in range(n)))[0]


class TestPairTable:
    def test_pairs(self):
        assert rp.pair_table("((.))") == [4, 3, -1, 1, 0]
        assert rp.pair_list("((.))..()") == [(0, 4), (1, 3), (7, 8)]

    @pytest.mark.parametrize("bad", ["((.)", "(.))", "((.]]", "(.x)"])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            rp.pair_table(bad)


class TestFlatCoordinates:
    def test_exterior_on_baseline_helices_above(self):
        db = "..((((...))))...((((....))))."
        xy = rp.flat_coordinates(db)
        ext = [
            k
            for k, j in enumerate(rp.pair_table(db))
            if j < 0 and not any(i < k < j for i, j in rp.pair_list(db))
        ]
        assert np.allclose(xy[ext, 1], 0.0)
        assert (xy[:, 1] >= -1e-9).all()
        assert (np.diff(xy[ext, 0]) > 0).all()  # exterior read 5'->3' left to right

    def test_unpaired_only(self):
        xy = rp.flat_coordinates("....")
        assert np.allclose(xy, [[0, 0], [1, 0], [2, 0], [3, 0]])

    def test_helices_do_not_overlap_horizontally(self):
        db = "((((....))))((((....))))"
        xy = rp.flat_coordinates(db)
        assert xy[:12, 0].max() < xy[12:, 0].min()

    def test_engines(self):
        for engine in ("naview", "simple", "puzzler", "auto"):
            assert rp.flat_coordinates("((((....))))", engine=engine).shape == (12, 2)

    @pytest.mark.parametrize("n,seed", [(76, 0), (150, 0), (150, 2), (400, 0), (400, 2), (1000, 0)])
    def test_auto_has_no_collisions(self, n, seed):
        # naview alone overlaps on these (checked when the fallback was added)
        db = _random_fold(n, seed)
        xy = rp.flat_coordinates(db)
        assert rp.min_distance(xy) >= 0.5
        assert rp.segment_crossings(xy, db) == 0

    def test_trna_clean(self):
        db = RNA.fold(TRNA_PHE)[0]
        xy = rp.flat_coordinates(db)
        assert rp.min_distance(xy) >= 0.6
        assert rp.segment_crossings(xy, db) == 0

    def test_global_plot_type_restored(self):
        old = RNA.cvar.rna_plot_type
        rp.flat_coordinates(_random_fold(200, 1), engine="puzzler")
        assert RNA.cvar.rna_plot_type == old


class TestSegmentCrossings:
    def test_cross(self):
        xy = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)  # 0-1 crosses 2-3
        assert rp.segment_crossings(xy, "....") == 1


class TestHairpinLoops:
    def test_sizes(self):
        assert rp.hairpin_loops("((...))..((((....))))") == [(2, 5), (13, 17)]
        assert rp.hairpin_loops("((...))..((((....))))", sizes=[4]) == [(13, 17)]
        assert rp.hairpin_loops("((.((...)).))") == [(5, 8)]


class TestColors:
    def test_pairing_probability_range(self):
        seq = "GGGGAAACCCC"
        p = rp.pairing_probability(seq, RNA.fold(seq)[0])
        assert p.shape == (11,)
        assert ((p >= 0) & (p <= 1)).all()

    def test_covariation(self):
        ref = "GG-AAACC"
        aln = {"a": ref, "b": "GA-AAAUC", "c": "CG-AAACG"}
        colors = rp.covariation_colors("((...))", aln, ref)
        assert len(colors) == 7
        assert colors[2] == (1.0, 1.0, 1.0, 1.0)  # unpaired: white
        assert colors[0] == colors[6]

    def test_covariation_length_mismatch(self):
        with pytest.raises(ValueError):
            rp.covariation_colors("((...))", ["GGAAACC"], "GGAAAC")


class TestDraw:
    def test_draw_all_annotations(self):
        seq = TRNA_PHE
        db = RNA.fold(seq)[0]
        fig, ax = plt.subplots(figsize=rp.suggest_figsize(rp.flat_coordinates(db)))
        res = rp.draw_structure(
            ax,
            seq,
            db,
            node_colors=rp.probability_colors(rp.pairing_probability(seq, db)),
            focus=(5, 70),
            loops=[rp.Loop(s, e, "hairpin loop") for s, e in rp.hairpin_loops(db)],
            tracks=[rp.Track(10, 20, "site A"), rp.Track(15, 25, "site B", "#e65100", layer=1)],
            constraint=db[:-1] + ".",
            highlight=[33],
            numbering=100,
        )
        labels = [h.get_label() for h in res.handles]
        assert {"site A", "site B", "hairpin loop", "flank", "highlighted nucleotide"} <= set(labels)
        assert labels.count("hairpin loop") == 1
        plt.close(fig)

    def test_numbering_labels_and_none(self):
        fig, ax = plt.subplots()
        rp.draw_structure(ax, "GGGAAACCC", "(((...)))", numbering=[1, 2, None, 4, 5, 6, 7, 8, 10])
        assert "10" in [t.get_text() for t in ax.texts]
        plt.close(fig)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tracks": [rp.Track(5, 20, "x")]},
            {"focus": (0, 0)},
            {"node_colors": ["white"]},
            {"numbering": [1, 2]},
        ],
    )
    def test_invalid_inputs(self, kwargs):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            rp.draw_structure(ax, "GGGAAACCC", "(((...)))", **kwargs)
        plt.close(fig)
