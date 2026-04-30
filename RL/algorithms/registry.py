"""Algorithm registry: maps a method key to (display name, adversary factory)."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from config import InnerConfig


# Factory signature: (cfg, device) -> adversary instance with `.adversarial_xi(...)`.
AdversaryFactory = Callable[[InnerConfig, torch.device], object]


def build_registry() -> Dict[str, Tuple[str, AdversaryFactory]]:
    from algorithms.algo1_wrm import WRMAdversary
    from algorithms.dual import DualAdversary
    from algorithms.icnn import ICNNAdversary
    from algorithms.new_ppa import NewPPAAdversary
    from algorithms.nn_dro import NNDROAdversary
    from algorithms.nominal import NominalAdversary
    from algorithms.npf import NPFAdversary
    from algorithms.particle import ParticleAdversary
    from algorithms.ppa import PPAAdversary
    from algorithms.rgo import RGOAdversary
    from algorithms.svg import SVGAdversary
    from algorithms.wfr import WFRAdversary
    from algorithms.wgf import WGFAdversary

    return {
        "nominal":  ("Nominal (SAA)",    lambda cfg, d: NominalAdversary(cfg, d)),
        "particle": ("Particle Ascent",  lambda cfg, d: ParticleAdversary(cfg, d)),
        "icnn":     ("ICNN / Brenier",   lambda cfg, d: ICNNAdversary(cfg, d)),
        "algo1":    ("Algo1 / WRM",      lambda cfg, d: WRMAdversary(cfg, d)),
        "npf":      ("NPF",              lambda cfg, d: NPFAdversary(cfg, d)),
        "ppa":      ("PPA",              lambda cfg, d: PPAAdversary(cfg, d)),
        "new_ppa":  ("New_PPA",          lambda cfg, d: NewPPAAdversary(cfg, d)),
        "dual":     ("Dual",             lambda cfg, d: DualAdversary(cfg, d)),
        "wgf":      ("WGF",              lambda cfg, d: WGFAdversary(cfg, d)),
        "wfr":      ("WFR",              lambda cfg, d: WFRAdversary(cfg, d)),
        "svg":      ("SVG",              lambda cfg, d: SVGAdversary(cfg, d)),
        "rgo":      ("RGO",              lambda cfg, d: RGOAdversary(cfg, d)),
        "nn_dro":   ("NN-DRO",           lambda cfg, d: NNDROAdversary(cfg, d)),
    }


def training_title(method: str, env_name: str) -> str:
    titles = {
        "nominal":  f"Nominal (vanilla PPO) — {env_name}",
        "particle": f"Particle Ascent — {env_name}",
        "icnn":     f"ICNN Transport (BB+Armijo) — {env_name}",
        "algo1":    f"Algo1 / WRM — {env_name}",
        "npf":      f"NPF (ICNN + BB+Armijo) — {env_name}",
        "ppa":      f"PPA — {env_name}",
        "new_ppa":  f"New_PPA — {env_name}",
        "dual":     f"Dual (log-sum-exp) — {env_name}",
        "wgf":      f"WGF (Langevin) — {env_name}",
        "wfr":      f"WFR — {env_name}",
        "svg":      f"SVG (Stein) — {env_name}",
        "rgo":      f"RGO — {env_name}",
        "nn_dro":   f"NN-DRO (vanilla MLP + Adam) — {env_name}",
    }
    return titles.get(method, f"{method} — {env_name}")
