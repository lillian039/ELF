"""Tests for the geometry router and geometry-routed attention.

Run with either:
    PYTHONPATH=src python -m unittest tests.test_geometry_router -v
    PYTHONPATH=src pytest -q tests/test_geometry_router.py   (if pytest installed)
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from configs.config import Config  # noqa: E402
from modules.geometry_router import (  # noqa: E402
    GeometryRouter, estimate_delta_rel, estimate_spherical_fit,
    geometry_model_kwargs, masked_token_subsample, pairwise_dist,
    parse_float_list, parse_layer_spec, validate_geometry_router_config,
)
from modules.layers import Attention, GeometryRoutedAttention  # noqa: E402
from modules.model import ELF_models  # noqa: E402


class TestParsers(unittest.TestCase):
    def test_parse_float_list(self):
        self.assertEqual(parse_float_list("0.25,0.5,1.0"), [0.25, 0.5, 1.0])

    def test_parse_layer_spec(self):
        self.assertEqual(parse_layer_spec("all", 4), {0, 1, 2, 3})
        self.assertEqual(parse_layer_spec("0,1,2", 12), {0, 1, 2})
        self.assertEqual(parse_layer_spec("0-3,6,8-11", 12),
                         {0, 1, 2, 3, 6, 8, 9, 10, 11})
        with self.assertRaises(ValueError):
            parse_layer_spec("0-13", 12)


class TestRouterShape(unittest.TestCase):
    """Test 1: router gates shape / normalization / finiteness."""

    def test_gates(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 16, 32)
        mask = torch.ones(2, 16)
        t = torch.tensor([0.1, 0.9])
        router = GeometryRouter()
        gates, scores = router(hidden, t, mask)
        self.assertEqual(gates.shape, (2, 3))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(gates).all())
        for key in ("e_H", "e_S", "logits", "gates"):
            self.assertTrue(torch.isfinite(scores[key]).all(), key)

    def test_requires_t(self):
        router = GeometryRouter()
        with self.assertRaises(ValueError):
            router(torch.randn(2, 16, 32), None, None)

    def test_learnable_bias_and_metrics(self):
        torch.manual_seed(0)
        router = GeometryRouter(learnable_bias=True, log_metrics=True)
        self.assertIn("bias", dict(router.named_parameters()))
        hidden = torch.randn(2, 16, 32)
        mask = torch.ones(2, 16)
        t = torch.tensor([0.1, 0.9])
        gates, _ = router(hidden, t, mask)
        self.assertEqual(gates.shape, (2, 3))
        self.assertIsNotNone(router.latest_gate_mean)
        self.assertIsNotNone(router.latest_e_h_mean)
        self.assertIsNotNone(router.latest_e_s_mean)
        self.assertIsNotNone(router.latest_logits_mean)


class TestDeltaRel(unittest.TestCase):
    """Test 2: points on a line are 0-hyperbolic -> small delta_rel."""

    def test_line_is_thin(self):
        pts = torch.arange(16, dtype=torch.float32).view(1, 16, 1) \
            * torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones(1, 16)
        D = pairwise_dist(pts, mask)
        delta_rel = estimate_delta_rel(D, mask, quad_samples=512)
        self.assertLess(delta_rel.item(), 0.1)

    def test_neutral_when_too_few_tokens(self):
        pts = torch.randn(1, 16, 4)
        mask = torch.zeros(1, 16)
        mask[0, :3] = 1  # only 3 valid tokens -> no valid quadruple
        D = pairwise_dist(pts, mask)
        delta_rel = estimate_delta_rel(D, mask, quad_samples=512)
        self.assertEqual(delta_rel.item(), 1.0)


class TestSphericalFit(unittest.TestCase):
    """Test 3: spherical fit is finite and in a sane range."""

    def test_random_no_crash(self):
        torch.manual_seed(1)
        pts = torch.randn(3, 16, 32)
        mask = torch.ones(3, 16)
        D = pairwise_dist(pts, mask)
        e_s = estimate_spherical_fit(D, mask, [0.25, 0.5, 1.0, 2.0, 4.0], rank_dim=32)
        self.assertTrue(torch.isfinite(e_s).all())
        self.assertTrue(((e_s >= 0) & (e_s <= 1)).all())

    def test_rank_penalty_fires(self):
        torch.manual_seed(1)
        pts = torch.randn(1, 16, 32)
        mask = torch.ones(1, 16)
        D = pairwise_dist(pts, mask)
        # rank cap of 2 cannot hold 16 random 32-d points -> nonzero residual.
        e_s = estimate_spherical_fit(D, mask, [0.25, 0.5, 1.0, 2.0, 4.0], rank_dim=1)
        self.assertGreater(e_s.item(), 0.0)

    def test_masked_tokens_ignored(self):
        torch.manual_seed(2)
        base = torch.randn(1, 16, 8)
        mask = torch.ones(1, 16)
        mask[0, 12:] = 0
        poisoned = base.clone()
        poisoned[0, 12:] = 1e3  # garbage in masked positions must not matter
        out = []
        for pts in (base, poisoned):
            s, sm = masked_token_subsample(pts, mask, 16)
            D = pairwise_dist(s, sm)
            out.append(estimate_spherical_fit(D, sm, [0.5, 1.0], rank_dim=8))
        self.assertTrue(torch.allclose(out[0], out[1]))


class TestGeometryRoutedAttention(unittest.TestCase):
    """Test 4: routed attention output shape / finiteness."""

    def test_forward(self):
        torch.manual_seed(0)
        B, N, C, H = 2, 12, 64, 4
        x = torch.randn(B, N, C)
        t = torch.tensor([0.2, 0.8])
        mask = torch.ones(B, N)
        for hyp in ("busemann_proxy", "poincare_distance"):
            for sph in ("cosine", "negative_angular"):
                attn = GeometryRoutedAttention(
                    C, H, geometry_router=GeometryRouter(),
                    hyperbolic_score=hyp, sphere_score=sph)
                out = attn(x, None, attention_mask=mask, t=t, geometry_mask=mask)
                self.assertEqual(out.shape, (B, N, C))
                self.assertTrue(torch.isfinite(out).all(), (hyp, sph))

    def test_fallback_matches_attention(self):
        torch.manual_seed(0)
        B, N, C, H = 2, 12, 64, 4
        x = torch.randn(B, N, C)
        mask = torch.ones(B, N)
        base = Attention(C, H)
        routed = GeometryRoutedAttention(C, H, geometry_router=None)
        routed.load_state_dict(base.state_dict())
        self.assertTrue(torch.allclose(
            base(x, None, attention_mask=mask),
            routed(x, None, attention_mask=mask, t=torch.zeros(B)),
            atol=1e-6,
        ))


class TestDisabledModelParity(unittest.TestCase):
    """Test 5: disabled geometry router keeps the original state_dict + forward."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=False,
        )

    def test_no_geometry_keys_in_state_dict(self):
        geo_keys = [k for k in self.model.state_dict() if "geometry" in k.lower()]
        self.assertEqual(geo_keys, [])
        for block in self.model.blocks:
            self.assertIsInstance(block.attn, Attention)

    def test_forward_smoke(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 512)
        t = torch.rand(2)
        # self_cond_cfg_scale is required whenever num_self_cond_cfg_tokens > 0
        # (the RoPE buffer reserves slots for those prefix tokens).
        sc = torch.ones(2)
        with torch.no_grad():
            out, dec = self.model(x, t, self_cond_cfg_scale=sc)
        self.assertEqual(out.shape, (2, 16, 512))
        self.assertIsNone(dec)
        self.assertTrue(torch.isfinite(out).all())

    def test_enabled_state_dict_unchanged(self):
        # With default fixed priors the router adds no learnable bias; the only
        # extra state_dict entries are the per-routed-layer gate_warmup_alpha
        # (a requires_grad=False Parameter, one per routed block). Everything
        # else must match the disabled model exactly.
        torch.manual_seed(0)
        routed = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0-3",
        )
        extra = set(routed.state_dict()) - set(self.model.state_dict())
        self.assertEqual(extra, {f"blocks.{i}.attn.geometry_router.gate_warmup_alpha"
                                 for i in range(4)})
        # gate_warmup_alpha must not be trainable.
        for i in range(4):
            self.assertFalse(routed.blocks[i].attn.geometry_router.gate_warmup_alpha.requires_grad)
        self.assertIsInstance(routed.blocks[0].attn, GeometryRoutedAttention)
        self.assertIsInstance(routed.blocks[4].attn, Attention)

    def test_learnable_bias_adds_router_params(self):
        torch.manual_seed(0)
        routed = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0,1",
            geometry_router_learnable_bias=True,
        )
        bias_keys = [k for k in routed.state_dict() if k.endswith("geometry_router.bias")]
        self.assertEqual(len(bias_keys), 2)


class TestEnabledModelForward(unittest.TestCase):
    """Enabled-router end-to-end forward, with mask and self-conditioning."""

    def test_forward(self):
        torch.manual_seed(0)
        model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0,1",
            geometry_router_sample_size=8, geometry_router_quad_samples=64,
        )
        x = torch.randn(2, 16, 512)
        t = torch.rand(2).clamp(0.05, 0.95)
        mask = torch.ones(2, 16)
        mask[1, 10:] = 0
        sc = torch.full((2,), 1.5)
        with torch.no_grad():
            out, dec = model(x, t, attention_mask=mask,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
        self.assertEqual(out.shape, (2, 16, 512))
        self.assertEqual(dec.shape, (2, 16, 128))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(dec).all())

    def test_metrics_collection(self):
        torch.manual_seed(0)
        model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0,1",
            geometry_router_sample_size=8, geometry_router_quad_samples=64,
            geometry_router_log_metrics=True,
        )
        x = torch.randn(2, 16, 512)
        t = torch.rand(2).clamp(0.05, 0.95)
        sc = torch.ones(2)
        with torch.no_grad():
            model(x, t, self_cond_cfg_scale=sc)
        metrics = model.geometry_router_metrics()
        self.assertEqual(set(metrics), {"layer_0", "layer_1"})
        for layer_metrics in metrics.values():
            for key in ("gate_e", "gate_h", "gate_s", "e_H", "e_S"):
                self.assertIn(key, layer_metrics)


class TestConfigValidation(unittest.TestCase):
    """Reserved options must be rejected loudly, not silently ignored."""

    def _enabled_config(self) -> Config:
        c = Config()
        c.geometry_router_enabled = True
        return c

    def test_disabled_config_never_validated(self):
        c = Config()
        c.geometry_router_on_attention = False  # nonsense, but router is off
        c.geometry_router_on_mlp = True
        validate_geometry_router_config(c)  # must not raise

    def test_on_attention_false_rejected(self):
        c = self._enabled_config()
        c.geometry_router_on_attention = False
        with self.assertRaises(NotImplementedError):
            geometry_model_kwargs(c)

    def test_on_mlp_true_rejected(self):
        c = self._enabled_config()
        c.geometry_router_on_mlp = True
        with self.assertRaises(NotImplementedError):
            geometry_model_kwargs(c)

    def test_hard_mode_rejected(self):
        c = self._enabled_config()
        c.geometry_router_mode = "hard"
        with self.assertRaises(NotImplementedError):
            geometry_model_kwargs(c)

    def test_valid_enabled_config_passes(self):
        kw = geometry_model_kwargs(self._enabled_config())
        self.assertTrue(kw["geometry_router_enabled"])
        self.assertFalse(kw["geometry_router_denoiser_only"])


class TestDenoiserOnly(unittest.TestCase):
    """geometry_router_denoiser_only: decoder rows / decode path stay Euclidean."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        common = dict(
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        )
        cls.disabled = ELF_models["ELF-B"](**common, geometry_router_enabled=False)
        # final_layer.linear is zero-initialized, which would make every
        # `output` identically zero and the difference assertions vacuous —
        # give it (shared) random weights so outputs are informative.
        torch.nn.init.normal_(cls.disabled.final_layer.linear.weight, std=0.02)
        torch.nn.init.normal_(cls.disabled.final_layer.linear.bias, std=0.02)
        # Hostile router prior (strongly hyperbolic) so routed rows visibly
        # differ from Euclidean rows.
        router_kw = dict(
            geometry_router_enabled=True, geometry_router_layers="all",
            geometry_router_sample_size=8, geometry_router_quad_samples=64,
            geometry_router_bias_e=-4.0, geometry_router_bias_h=4.0,
            geometry_router_time_e_bias=0.0,
        )
        # strict=False: routed models carry extra router params (learnable
        # bias + gate_warmup_alpha) absent from the disabled model; we only
        # want to copy the shared backbone weights so the arms match.
        cls.routed = ELF_models["ELF-B"](
            **common, **router_kw, geometry_router_denoiser_only=True)
        cls.routed.load_state_dict(cls.disabled.state_dict(), strict=False)
        cls.routed_always = ELF_models["ELF-B"](
            **common, **router_kw, geometry_router_denoiser_only=False)
        cls.routed_always.load_state_dict(cls.disabled.state_dict(), strict=False)
        for m in (cls.disabled, cls.routed, cls.routed_always):
            m.eval()
        torch.manual_seed(7)
        cls.x = torch.randn(2, 16, 512)
        cls.t = torch.tensor([0.3, 0.6])
        cls.sc = torch.ones(2)

    def test_decode_path_bypasses_routing(self):
        # decoder_step_active=True (python bool) = decode-mode forward:
        # denoiser-only model must match the disabled model exactly.
        with torch.no_grad():
            out_d, dec_d = self.disabled(self.x, self.t, self_cond_cfg_scale=self.sc,
                                         decoder_step_active=True)
            out_r, dec_r = self.routed(self.x, self.t, self_cond_cfg_scale=self.sc,
                                       decoder_step_active=True)
            out_a, _ = self.routed_always(self.x, self.t, self_cond_cfg_scale=self.sc,
                                          decoder_step_active=True)
        self.assertTrue(torch.equal(out_d, out_r))
        self.assertTrue(torch.equal(dec_d, dec_r))
        # ... while the default (shared-backbone) mode keeps routing here.
        self.assertFalse(torch.allclose(out_d, out_a, atol=1e-3))

    def test_denoiser_forward_stays_routed(self):
        # No decoder_step_active = pure denoiser call: routing must be live.
        with torch.no_grad():
            out_d, _ = self.disabled(self.x, self.t, self_cond_cfg_scale=self.sc)
            out_r, _ = self.routed(self.x, self.t, self_cond_cfg_scale=self.sc)
        self.assertFalse(torch.allclose(out_d, out_r, atol=1e-3))

    def test_mixed_rows(self):
        # Row 0 decoder-mode (forced Euclidean), row 1 denoiser-mode (routed).
        active = torch.tensor([1.0, 0.0])
        with torch.no_grad():
            out_d, _ = self.disabled(self.x, self.t, self_cond_cfg_scale=self.sc,
                                     decoder_step_active=active)
            out_r, _ = self.routed(self.x, self.t, self_cond_cfg_scale=self.sc,
                                   decoder_step_active=active)
        # Decoder row: Euclidean gate -> matches disabled model (custom
        # softmax vs SDPA gives tiny numerical差, hence atol not equal).
        self.assertTrue(torch.allclose(out_d[0], out_r[0], atol=1e-4))
        # Denoiser row: visibly routed away from Euclidean.
        self.assertFalse(torch.allclose(out_d[1], out_r[1], atol=1e-3))


if __name__ == "__main__":
    unittest.main()
