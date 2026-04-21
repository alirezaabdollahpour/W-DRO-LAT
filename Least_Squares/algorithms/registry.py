"""Registry mapping algorithm keys to (display name, solve_fn)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from algorithms.dual import solve_dual
from algorithms.erm import solve_erm_closed_form
from algorithms.icnn import solve_icnn_map
from algorithms.madry import solve_madry_pgd
from algorithms.nn_dro import solve_nn_dro
from algorithms.npf import solve_npf_icnn_map
from algorithms.particle_ascent import solve_Particle_Ascent
from algorithms.ppa import solve_ppa
from algorithms.wfr import solve_wfr
from algorithms.wgf import solve_wgf


# Algorithms whose solver returns more than just theta (theta, psi[, diagnostics]).
EXTENDED_RETURN_ALGORITHMS = {"icnn", "npf"}


RegistryEntry = Tuple[str, Callable[..., Any]]


def build_registry() -> Dict[str, RegistryEntry]:
    return {
        "erm": ("ERM", solve_erm_closed_form),
        "particle_ascent": ("Particle Ascent", solve_Particle_Ascent),
        "wgf": ("WGF(Otto, 1996)", solve_wgf),
        "wfr": ("WFR(Xu, 2025)", solve_wfr),
        "dual": ("Dual(Wang et al., 2021)", solve_dual),
        "icnn": ("ICNN", solve_icnn_map),
        "madry": ("Madry PGD", solve_madry_pgd),
        "ppa": ("PPA", solve_ppa),
        "npf": ("WDRO-NPF", solve_npf_icnn_map),
        "nn_dro": ("NN-DRO", solve_nn_dro),
    }
