"""Registry mapping algorithm keys to (display name, train fn, checkpoint fn)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from algorithms.algo1_wrm import train_algorithm_1
from algorithms.algo2_icnn import train_algorithm_2
from algorithms.dual import train_algorithm_dual
from algorithms.new_ppa import train_algorithm_new_ppa
from algorithms.npf import train_algorithm_npf
from algorithms.ppa import train_algorithm_ppa
from algorithms.rgo import train_algorithm_rgo
from algorithms.saa import train_algorithm_saa
from algorithms.svg import train_algorithm_svg
from algorithms.wfr import train_algorithm_wfr
from algorithms.wgf import train_algorithm_wgf
from utils.checkpoints import (
    save_checkpoint_algo1,
    save_checkpoint_algo2,
    save_checkpoint_new_ppa,
    save_checkpoint_npf,
    save_checkpoint_ppa,
    save_checkpoint_simple,
)

# Algorithms that return (state, icnn_state, logs) instead of (state, logs).
DOUBLE_STATE_ALGORITHMS = {"algo2", "npf"}


RegistryEntry = Tuple[str, Callable[..., Any], Callable[..., str]]


def build_registry() -> Dict[str, RegistryEntry]:
    return {
        "saa": (
            "SAA", train_algorithm_saa,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "saa", "SAA"),
        ),
        "algo1": ("Algo1 / WRM", train_algorithm_1, save_checkpoint_algo1),
        "algo2": ("Algo2 / ICNN", train_algorithm_2, save_checkpoint_algo2),
        "npf": ("NPF / ICNN-NPF", train_algorithm_npf, save_checkpoint_npf),
        "ppa": ("PPA", train_algorithm_ppa, save_checkpoint_ppa),
        "new_ppa": ("New_PPA", train_algorithm_new_ppa, save_checkpoint_new_ppa),
        "dual": (
            "Dual", train_algorithm_dual,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "dual", "Dual"),
        ),
        "wgf": (
            "WGF", train_algorithm_wgf,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "wgf", "WGF"),
        ),
        "wfr": (
            "WFR", train_algorithm_wfr,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "wfr", "WFR"),
        ),
        "svg": (
            "SVG", train_algorithm_svg,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "svg", "SVG"),
        ),
        "rgo": (
            "RGO", train_algorithm_rgo,
            lambda st, cfg_: save_checkpoint_simple(st, cfg_, "rgo", "RGO"),
        ),
    }
