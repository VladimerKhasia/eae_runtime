import torch

from eae_runtime import AdjointState, AdjointPipeline, EventBus
from eae_runtime.passes import (
    ClipPass,
    FP16Pass,
    FP8Pass,
    GaussianNoisePass,
    Int8QuantizationPass,
    LoggingPass,
    RegularizationPass,
    SyntheticGradientPass,
)


def _adj(t):
    return AdjointState(tensor=t, layer_id=1, block="b")


def test_clip_pass_reduces_norm_above_threshold():
    t = torch.ones(10) * 5.0  # norm = sqrt(10)*5 ~ 15.8
    p = ClipPass(max_norm=1.0)
    out = p(_adj(t))
    assert out.norm().item() <= 1.0 + 1e-4


def test_clip_pass_leaves_small_norm_untouched():
    t = torch.ones(4) * 0.01
    p = ClipPass(max_norm=10.0)
    out = p(_adj(t))
    assert torch.allclose(out.tensor, t)


def test_fp16_pass_introduces_bounded_error():
    t = torch.randn(100) * 1000
    p = FP16Pass()
    out = p(_adj(t))
    assert out.tensor.dtype == t.dtype
    assert not torch.equal(out.tensor, t)  # precision was actually lost
    assert torch.allclose(out.tensor, t, rtol=1e-2)


def test_int8_quantization_pass_reduces_precision_but_preserves_sign():
    t = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0])
    p = Int8QuantizationPass()
    out = p(_adj(t))
    assert torch.equal(torch.sign(out.tensor), torch.sign(t))
    assert "quantization_scale" in out.metadata


def test_int8_quantization_handles_all_zero_tensor():
    t = torch.zeros(5)
    p = Int8QuantizationPass()
    out = p(_adj(t))
    assert torch.equal(out.tensor, t)


def test_fp8_pass_is_a_valid_alias():
    t = torch.randn(8)
    out = FP8Pass()(_adj(t))
    assert out.tensor.shape == t.shape


def test_regularization_pass_shrinks_adjoint():
    t = torch.ones(4)
    p = RegularizationPass(strength=0.1)
    out = p(_adj(t))
    assert torch.allclose(out.tensor, t * 0.9)


def test_regularization_pass_noop_at_zero_strength():
    t = torch.randn(4)
    p = RegularizationPass(strength=0.0)
    out = p(_adj(t))
    assert torch.allclose(out.tensor, t)


def test_gaussian_noise_pass_changes_tensor_but_preserves_shape():
    gen = torch.Generator().manual_seed(0)
    t = torch.zeros(1000)
    p = GaussianNoisePass(std=1.0, generator=gen)
    out = p(_adj(t))
    assert out.tensor.shape == t.shape
    assert out.tensor.std().item() > 0.5  # roughly std=1 noise was added


def test_gaussian_noise_noop_at_zero_std():
    t = torch.randn(4)
    p = GaussianNoisePass(std=0.0)
    out = p(_adj(t))
    assert torch.allclose(out.tensor, t)


def test_logging_pass_emits_event_and_does_not_mutate():
    bus = EventBus()
    bus.start_recording()
    p = LoggingPass(event_bus=bus)
    t = torch.randn(4)
    out = p(_adj(t))
    assert torch.allclose(out.tensor, t)
    events = bus.events_of_type("AdjointModified")
    assert len(events) == 1
    assert "mean" in events[0].payload


def test_synthetic_gradient_pass_trains_predictor_and_can_replace_adjoint():
    torch.manual_seed(0)
    sg = SyntheticGradientPass(feature_dim=8, warmup_steps=0, use_synthetic=True)
    t = torch.randn(2, 8)
    out = sg(_adj(t))
    assert out.tensor.shape == t.shape
    assert sg.last_loss is not None


def test_pipeline_applies_passes_in_order_and_records_history():
    pipeline = AdjointPipeline(passes=[RegularizationPass(0.5), ClipPass(max_norm=100.0)])
    t = torch.ones(4) * 2.0
    out = pipeline.run(_adj(t))
    assert torch.allclose(out.tensor, t * 0.5)
    assert out.history == ["RegularizationPass", "ClipPass"]


def test_empty_pipeline_is_identity():
    pipeline = AdjointPipeline(passes=[])
    t = torch.randn(4)
    out = pipeline.run(_adj(t))
    assert torch.allclose(out.tensor, t)


def test_pipeline_emits_pass_applied_events():
    bus = EventBus()
    bus.start_recording()
    pipeline = AdjointPipeline(passes=[ClipPass(1.0)], event_bus=bus)
    pipeline.run(_adj(torch.ones(4) * 10))
    events = bus.events_of_type("PassApplied")
    assert len(events) == 1
    assert events[0].payload["pass_name"] == "ClipPass"
