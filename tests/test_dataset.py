import random

import pytest

import cv_content as C
from dataset import (achievement_density, classify_ad, count_achievements, generate_cv_lines, generate_dataset,
                     is_achievement, load_manifest, validate, word_count)


@pytest.mark.parametrize("s", [
    "Increased newsletter open rates by 23%.",
    "Saved $45K per year by renegotiating supplier contracts.",
    "Onboarded 1,200 users in the first quarter.",
    "Delivered the migration within 6 months.",
    "Led a team of 8 engineers.",
    "Received the Innovation Award for the project.",
    "Holds a patent on sensor calibration.",
    "Doubled qualified leads from partner channels.",
])
def test_achievement_patterns_match(s):
    assert is_achievement(s)


@pytest.mark.parametrize("s", [
    "Supported the weekly release process with attention to detail.",
    "Junior Software Developer — Northwind Labs (2021 – 2023)",
    "BSc in Computer Science — Westbrook College (2019)",
    "Brings over 10 years of experience in retail.",
    "Python, SQL, Git, REST APIs.",
])
def test_non_achievements(s):
    assert not is_achievement(s)


def test_duty_vocabulary_never_counts_as_achievement():
    for key, spec in C.ROLE_SPECS.items():
        for v in C.DUTY_VERBS:
            for o in spec["duty_objects"]:
                for c in C.DUTY_CONTEXTS:
                    assert not is_achievement(f"{v} {o} {c}."), (key, o, c)
    for s in C.SUMMARY_SENTENCES:
        assert not is_achievement(s)


def test_density_formula():
    text = "Increased sales by 20%. " + " ".join(["word"] * 96)
    assert word_count(text) == 100
    assert count_achievements(text) == 1
    assert achievement_density(text) == pytest.approx(1.0)


@pytest.mark.parametrize("score,cls", [(2.0, "High"), (1.5, "High"), (1.49, "Medium"), (0.51, "Medium"),
                                       (0.5, "Low"), (0.0, "Low")])
def test_classify_thresholds(score, cls):
    assert classify_ad(score) == cls


def test_generated_cv_hits_its_cell():
    for role in ("jr_developer", "project_manager", "professor"):
        for ad in ("High", "Medium", "Low"):
            lines = generate_cv_lines(role, ad, "Test Person", random.Random(1))
            text = "\n".join(lines)
            assert lines[0] == "Test Person"
            assert classify_ad(achievement_density(text)) == ad


def test_full_dataset(tmp_path):
    recs = generate_dataset(tmp_path, seed=7, make_pdf=False)
    assert len(recs) == 27
    assert len({r.name for r in recs}) == 27
    cells = {(r.seniority, r.ad_class) for r in recs}
    assert len(cells) == 9
    assert len(load_manifest(tmp_path)) == 27
    assert validate(tmp_path) == []
