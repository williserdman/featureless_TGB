from __future__ import annotations

import importlib
import unittest

import torch
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset


def count_label_batches_for_split(
    dataset: PyGNodePropPredDataset,
    data,
    split_mask: torch.Tensor,
    *,
    max_events: int | None,
) -> tuple[int, int, int]:
    """Replay label-consumption logic used in main.py without model updates.

    Returns:
    - number of processed events
    - number of events that yielded at least one label batch
    - total number of consumed label batches
    """

    dataset.reset_label_time()
    edge_idx = torch.where(split_mask)[0]
    if max_events is not None:
        edge_idx = edge_idx[:max_events]

    labeled_events = 0
    total_label_batches = 0

    for idx in edge_idx.tolist():
        current_time = int(data.t[idx].item())
        batches_this_event = 0
        while True:
            label_tuple = dataset.get_node_label(current_time)
            if label_tuple is None:
                break
            batches_this_event += 1

        if batches_this_event > 0:
            labeled_events += 1
        total_label_batches += batches_this_event

    return int(edge_idx.numel()), labeled_events, total_label_batches


class TestProjectImports(unittest.TestCase):
    def test_core_modules_import(self) -> None:
        modules = [
            "main",
            "models",
            "sampling",
            "feature_modes",
            "synthetic_features",
            "metrics_utils",
            "plot_results",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


class TestMaskingAndLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trade_ds = PyGNodePropPredDataset(name="tgbn-trade", root="datasets")
        cls.trade_data = cls.trade_ds.get_TemporalData()

        cls.genre_ds = PyGNodePropPredDataset(name="tgbn-genre", root="datasets")
        cls.genre_data = cls.genre_ds.get_TemporalData()

    def test_masks_are_disjoint(self) -> None:
        for name, ds in [("tgbn-trade", self.trade_ds), ("tgbn-genre", self.genre_ds)]:
            overlap_tv = int(torch.logical_and(ds.train_mask, ds.val_mask).sum().item())
            overlap_tt = int(torch.logical_and(ds.train_mask, ds.test_mask).sum().item())
            overlap_vt = int(torch.logical_and(ds.val_mask, ds.test_mask).sum().item())
            self.assertEqual(overlap_tv, 0, f"{name}: train/val masks overlap")
            self.assertEqual(overlap_tt, 0, f"{name}: train/test masks overlap")
            self.assertEqual(overlap_vt, 0, f"{name}: val/test masks overlap")

    def test_masks_are_temporally_ordered(self) -> None:
        for name, ds, data in [
            ("tgbn-trade", self.trade_ds, self.trade_data),
            ("tgbn-genre", self.genre_ds, self.genre_data),
        ]:
            train_t = data.t[ds.train_mask]
            val_t = data.t[ds.val_mask]
            test_t = data.t[ds.test_mask]

            self.assertTrue(
                bool(train_t.max() <= val_t.min()),
                f"{name}: expected max(train_t) <= min(val_t)",
            )
            self.assertTrue(
                bool(val_t.max() <= test_t.min()),
                f"{name}: expected max(val_t) <= min(test_t)",
            )

    def test_trade_train_cap_effect_on_labels(self) -> None:
        e_5k, labeled_5k, batches_5k = count_label_batches_for_split(
            self.trade_ds,
            self.trade_data,
            self.trade_ds.train_mask,
            max_events=5000,
        )
        e_20k, labeled_20k, batches_20k = count_label_batches_for_split(
            self.trade_ds,
            self.trade_data,
            self.trade_ds.train_mask,
            max_events=20000,
        )

        self.assertEqual(e_5k, 5000)
        self.assertEqual(
            batches_5k,
            0,
            "tgbn-trade first 5000 train events should have 0 label batches; this explains train NaN.",
        )
        self.assertGreater(
            batches_20k,
            0,
            "tgbn-trade should expose train labels when max-events is increased.",
        )
        self.assertGreaterEqual(labeled_20k, labeled_5k)

    def test_genre_train_has_labels_with_5k_cap(self) -> None:
        e_5k, labeled_5k, batches_5k = count_label_batches_for_split(
            self.genre_ds,
            self.genre_data,
            self.genre_ds.train_mask,
            max_events=5000,
        )
        self.assertEqual(e_5k, 5000)
        self.assertGreater(
            batches_5k,
            0,
            "tgbn-genre typically has some label batches in first 5000 train events.",
        )
        self.assertGreater(labeled_5k, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
