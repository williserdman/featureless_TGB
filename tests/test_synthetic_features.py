from __future__ import annotations

import unittest

import torch

from synthetic_features import FeatureEngineConfig, NodeFeatureConfig, NodeFeatureGenerator, SyntheticFeatureEngine


class TestSyntheticFeatureEngine(unittest.TestCase):
    def test_full_message_is_unchanged(self) -> None:
        cfg = FeatureEngineConfig(
            mode="full",
            num_nodes=10,
            feature_dim=8,
            noise_seed=123,
            noise_std=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)
        base_msg = torch.randn((1, 4), dtype=torch.float32)
        out = engine.event_message(event_index=0, src=1, dst=2, current_time=1, base_msg=base_msg)
        self.assertTrue(torch.allclose(out, base_msg))

    def test_unweighted_ones_returns_constant_tensor(self) -> None:
        cfg = FeatureEngineConfig(
            mode="unweighted_ones",
            num_nodes=10,
            feature_dim=6,
            noise_seed=123,
            noise_std=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)
        base_msg = torch.zeros((1, 4), dtype=torch.float32)
        out = engine.event_message(event_index=0, src=1, dst=2, current_time=1, base_msg=base_msg)
        self.assertEqual(out.shape, (1, 6))
        self.assertTrue(torch.all(out == 1.0))

    def test_gaussian_noise_is_deterministic_by_event_index(self) -> None:
        cfg = FeatureEngineConfig(
            mode="gaussian_noise",
            num_nodes=10,
            feature_dim=8,
            noise_seed=123,
            noise_std=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)

        base_msg = torch.zeros((1, 4), dtype=torch.float32)
        m1 = engine.event_message(event_index=7, src=1, dst=2, current_time=1, base_msg=base_msg)
        engine.reset()
        m2 = engine.event_message(event_index=7, src=1, dst=2, current_time=1, base_msg=base_msg)
        self.assertTrue(torch.allclose(m1, m2))

    def test_temporal_delta_is_causal(self) -> None:
        cfg = FeatureEngineConfig(
            mode="temporal_delta",
            num_nodes=10,
            feature_dim=12,
            noise_seed=5,
            noise_std=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)
        base_msg = torch.zeros((1, 4), dtype=torch.float32)

        first = engine.event_message(event_index=0, src=3, dst=4, current_time=10, base_msg=base_msg)
        engine.update_state(src=3, dst=4, current_time=10)
        second = engine.event_message(event_index=1, src=3, dst=4, current_time=15, base_msg=base_msg)

        self.assertEqual(first.shape, (1, 12))
        self.assertEqual(second.shape, (1, 12))
        self.assertFalse(torch.allclose(first, second))

    def test_node_gaussian_features_deterministic_per_time(self) -> None:
        cfg = NodeFeatureConfig(
            mode="gaussian_noise",
            num_nodes=8,
            node_feature_dim=12,
            noise_seed=71,
            noise_std=0.5,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
            gae_refresh_interval=4,
            gae_steps=2,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        g1 = NodeFeatureGenerator(cfg)
        g2 = NodeFeatureGenerator(cfg)

        f1 = g1.node_features_for_time(current_time=123)
        f2 = g2.node_features_for_time(current_time=123)
        self.assertTrue(torch.allclose(f1, f2))

    def test_node_snapshot_pagerank_is_causal(self) -> None:
        cfg = NodeFeatureConfig(
            mode="snapshot_pagerank",
            num_nodes=8,
            node_feature_dim=10,
            noise_seed=31,
            noise_std=1.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
            gae_refresh_interval=4,
            gae_steps=2,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        gen = NodeFeatureGenerator(cfg)
        f0 = gen.node_features_for_time(current_time=10)
        gen.update_state(src=1, dst=2, current_time=10)
        f1 = gen.node_features_for_time(current_time=11)
        self.assertEqual(f0.shape, (8, 10))
        self.assertEqual(f1.shape, (8, 10))
        self.assertFalse(torch.allclose(f0, f1))

    def test_node_snapshot_node2vec_is_causal_and_fixed_dim(self) -> None:
        cfg = NodeFeatureConfig(
            mode="snapshot_node2vec",
            num_nodes=12,
            node_feature_dim=10,
            noise_seed=77,
            noise_std=1.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=0.5,
            node2vec_q=2.0,
            gae_refresh_interval=4,
            gae_steps=2,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        gen = NodeFeatureGenerator(cfg)
        base = gen.node_features_for_time(current_time=10)
        gen.update_state(src=1, dst=2, current_time=10)
        updated = gen.node_features_for_time(current_time=11)
        self.assertEqual(base.shape, (12, 10))
        self.assertEqual(updated.shape, (12, 10))
        self.assertFalse(torch.allclose(base, updated))

    def test_node_snapshot_deepwalk_is_deterministic(self) -> None:
        cfg = NodeFeatureConfig(
            mode="snapshot_deepwalk",
            num_nodes=12,
            node_feature_dim=10,
            noise_seed=99,
            noise_std=1.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
            gae_refresh_interval=4,
            gae_steps=2,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        g1 = NodeFeatureGenerator(cfg)
        g2 = NodeFeatureGenerator(cfg)
        g1.update_state(src=1, dst=2, current_time=10)
        g2.update_state(src=1, dst=2, current_time=10)
        out1 = g1.node_features_for_time(current_time=11)
        out2 = g2.node_features_for_time(current_time=11)
        self.assertTrue(torch.allclose(out1, out2))

    def test_node_snapshot_gae_is_deterministic(self) -> None:
        cfg = NodeFeatureConfig(
            mode="snapshot_gae",
            num_nodes=12,
            node_feature_dim=8,
            noise_seed=111,
            noise_std=1.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
            gae_refresh_interval=2,
            gae_steps=3,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        g1 = NodeFeatureGenerator(cfg)
        g2 = NodeFeatureGenerator(cfg)
        for src, dst in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            g1.update_state(src=src, dst=dst, current_time=10)
            g2.update_state(src=src, dst=dst, current_time=10)

        out1 = g1.node_features_for_time(current_time=11)
        out2 = g2.node_features_for_time(current_time=11)
        self.assertEqual(out1.shape, (12, 8))
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_node_snapshot_gae_refresh_interval_reuses_cached_features(self) -> None:
        cfg = NodeFeatureConfig(
            mode="snapshot_gae",
            num_nodes=10,
            node_feature_dim=8,
            noise_seed=123,
            noise_std=1.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
            gae_refresh_interval=3,
            gae_steps=2,
            gae_lr=0.05,
            gae_max_edges=256,
            gae_batch_size=32,
        )
        gen = NodeFeatureGenerator(cfg)
        gen.update_state(src=1, dst=2, current_time=1)
        first = gen.node_features_for_time(current_time=2)

        gen.update_state(src=2, dst=3, current_time=3)
        second = gen.node_features_for_time(current_time=4)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

        gen.update_state(src=3, dst=4, current_time=5)
        gen.update_state(src=4, dst=5, current_time=6)
        third = gen.node_features_for_time(current_time=7)
        self.assertFalse(torch.allclose(second, third, atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
