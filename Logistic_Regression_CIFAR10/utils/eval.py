"""Clean + PGD-L2 evaluation over a list of attack radii."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def evaluate_model(
    model_object,
    test_features_np: np.ndarray,
    test_labels_np: np.ndarray,
    attack_fn: Callable,
    epsilon_list: List[float],
    device: torch.device,
    eval_batch_size: int = 128,
    pgd_num_iter: int = 10,
) -> Dict[float, float]:
    """Clean accuracy at ε=0 and PGD accuracy at each ε in ``epsilon_list``."""
    pytorch_model = model_object.model.eval()

    clean_accuracy = model_object.score(test_features_np, test_labels_np) * 100
    results: Dict[float, float] = {0.0: clean_accuracy}
    print(f"Accuracy on clean test set: {results[0.0]:.2f}%")

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(test_features_np).float(),
            torch.from_numpy(test_labels_np).long(),
        ),
        batch_size=eval_batch_size,
    )

    for epsilon in epsilon_list:
        if epsilon == 0.0:
            continue
        correct, total = 0, 0
        for features, labels in tqdm(
            test_loader, desc=f"Evaluating (epsilon={epsilon:.4f})", leave=False
        ):
            perturbed_features = attack_fn(
                model=pytorch_model,
                features=features,
                labels=labels,
                epsilon=epsilon,
                alpha=epsilon / 8,
                num_iter=pgd_num_iter,
                device=device,
            )
            with torch.no_grad():
                outputs = pytorch_model(perturbed_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        results[epsilon] = 100.0 * correct / total
        print(f"Accuracy under PGD attack (epsilon={epsilon:.4f}): {results[epsilon]:.2f}%")
    return results
