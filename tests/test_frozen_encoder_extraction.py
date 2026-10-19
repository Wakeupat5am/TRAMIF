import unittest
import copy
import json
from unittest.mock import patch
import numpy as np
import torch
from models.malsbslcnet import MalSBSLCNet
from models.compact_view_net import CompactViewNet
from scripts.frozen_encoder_extraction import features, state_hash, aligned, plan, VIEWS, verify_config_equivalence
from scripts.mmd_measurement_preparation import checked_rows


class TestFrozenExtraction(unittest.TestCase):
    def test_checkpoint_json_tuple_list_roundtrip(self):
        embedded = {'adam_defaults': {'betas': (0.9, 0.999)},
                    'dataset': {'num_classes': 32, 'cutoff_exclusive': '2019-11',
                                'class_to_idx': {'a': 0, 'b': 1}}}
        saved = json.loads(json.dumps(embedded))
        self.assertNotEqual(embedded, saved)
        verify_config_equivalence(embedded, saved)

    def test_config_comparison_rejects_semantic_changes(self):
        original = {'betas': (0.9, 0.999), 'K': 32, 'cutoff': '2019-11',
                    'mapping': {'a': 0, 'b': 1}}
        for key, value in [('betas', [0.9, 0.99]), ('K', 42), ('K', '32'),
                           ('cutoff', '2019-12'), ('mapping', {'a': 1, 'b': 0})]:
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                verify_config_equivalence(original, changed)
        with self.assertRaises(ValueError):
            verify_config_equivalence(original, {**original, 'extra': True})
        with self.assertRaises(ValueError):
            verify_config_equivalence({'value': float('nan')}, {'value': float('nan')})

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_endpoints_shapes_state_and_repeatability(self):
        torch.manual_seed(42)
        batch = torch.rand(3, 1, 64, 64)
        for k in (32, 42, 51):
            for view in VIEWS:
                model = (MalSBSLCNet(k) if view == 'sbsmi' else CompactViewNet(k)).eval().requires_grad_(False)
                before = state_hash(model)
                actual = features(model, view, batch)
                with torch.inference_mode():
                    # Independent explicit traversal of the specified endpoint.
                    expected = model.head_norm(model.flatten(model.pool(model.blocks(model.stem(batch))))).numpy()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, features(model, view, batch))
                self.assertEqual(actual.shape, (3, 2048 if view == 'sbsmi' else 512))
                self.assertEqual(before, state_hash(model))
                self.assertFalse(model.head_norm._forward_hooks)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_train_mode_and_unfrozen_rejected(self):
        model = MalSBSLCNet(32)
        batch = torch.zeros(2, 1, 64, 64)
        with self.assertRaises(ValueError): features(model, 'sbsmi', batch)
        model.eval()
        with self.assertRaises(ValueError): features(model, 'sbsmi', batch)

    def test_nonfinite_rejected(self):
        model = CompactViewNet(42).eval().requires_grad_(False)
        with self.assertRaises(ValueError):
            features(model, 'entropy', torch.full((2, 1, 64, 64), float('nan')))

    def test_hook_removed_on_failure(self):
        model = MalSBSLCNet(32).eval().requires_grad_(False)
        with patch.object(model.classifier, 'forward', side_effect=RuntimeError('synthetic')):
            with self.assertRaises(RuntimeError): features(model, 'sbsmi', torch.zeros(2, 1, 64, 64))
        self.assertFalse(model.head_norm._forward_hooks)

    def test_alignment_rejects_permutation_duplicate_missing(self):
        aligned(['a', 'b'], {v: ['a', 'b'] for v in VIEWS})
        for observed in ({v: ['b', 'a'] if v == 'entropy' else ['a', 'b'] for v in VIEWS}, {}):
            with self.assertRaises(ValueError): aligned(['a', 'b'], observed)
        with self.assertRaises(ValueError): aligned(['a', 'a'], {v: ['a', 'a'] for v in VIEWS})

    def test_forbidden_stage_fails_before_io(self):
        with self.assertRaises(ValueError): plan('E1', 'C', None)
        with self.assertRaises(ValueError): plan('Future', 'A', None)

    def test_chronology_guard(self):
        row = dict(sha='a'*64, family='test', timestamp='2020-04-01T00:00:00+00:00', stage='A')
        with self.assertRaises(ValueError): checked_rows([row], 'Final', ['test'], {'A'})
        row.update(timestamp='2019-12-01T00:00:00+00:00', stage='train')
        with self.assertRaises(ValueError): checked_rows([row], 'E1', ['test'], {'train'})


if __name__ == '__main__': unittest.main()
