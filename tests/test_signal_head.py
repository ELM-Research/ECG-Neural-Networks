import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from neural_networks.transformer.discrete.decoder import DecoderTransformerConfig, DecoderTransformer
from neural_networks.transformer.discrete.signal_head import (
    SignalFlowHeadConfig, SignalFlowHead, SignalFlowHeadOutput, DecoderWithSignalHead, timestep_embedding,
)


@pytest.fixture
def signal_head():
    cfg = SignalFlowHeadConfig(signal_dim=12, d_model=64, n_heads=4, dim_ff=128, num_layers=2, max_signal_len=100, num_steps=10)
    return SignalFlowHead(cfg)


@pytest.fixture
def decoder():
    cfg = DecoderTransformerConfig(vocab_size=100, pad_id=0, d_model=64, n_heads=4, dim_ff=128, num_layers=2, max_seq_len=64)
    return DecoderTransformer(cfg)


def test_timestep_embedding_shape():
    t = torch.rand(4)
    emb = timestep_embedding(t, 64)
    assert emb.shape == (4, 64)


def test_timestep_embedding_odd_dim():
    emb = timestep_embedding(torch.rand(2), 65)
    assert emb.shape == (2, 65)


def test_signal_flow_head_forward(signal_head):
    bsz, c, seq_len, ctx_len = 2, 12, 100, 20
    context = torch.randn(bsz, ctx_len, 64)
    signal = torch.randn(bsz, c, seq_len)
    out = signal_head(context, signal)
    assert isinstance(out, SignalFlowHeadOutput)
    assert out.loss.shape == ()
    assert out.loss.item() >= 0
    assert out.prediction.shape == (bsz, c, seq_len)


def test_signal_flow_head_forward_with_mask(signal_head):
    bsz, c, seq_len, ctx_len = 2, 12, 100, 20
    context = torch.randn(bsz, ctx_len, 64)
    context_mask = torch.zeros(bsz, ctx_len, dtype=torch.bool)
    context_mask[:, -5:] = True
    signal = torch.randn(bsz, c, seq_len)
    out = signal_head(context, signal, context_mask=context_mask)
    assert out.loss.shape == ()
    assert out.prediction.shape == (bsz, c, seq_len)


def test_signal_flow_head_sample(signal_head):
    bsz, c, seq_len, ctx_len = 2, 12, 100, 20
    context = torch.randn(bsz, ctx_len, 64)
    signal_head.eval()
    sampled = signal_head.sample(context, (bsz, c, seq_len), torch.device("cpu"), num_steps=5)
    assert sampled.shape == (bsz, c, seq_len)


def test_signal_flow_head_zero_init(signal_head):
    assert (signal_head.output_proj.weight == 0).all()
    assert (signal_head.output_proj.bias == 0).all()


def test_decoder_with_signal_head_forward(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=False)
    bsz = 2
    tgt_ids = torch.randint(1, 100, (bsz, 32))
    signal = torch.randn(bsz, 12, 100)
    labels = tgt_ids.clone()
    out = model(tgt_ids, signal, labels=labels)
    assert out.loss.shape == ()
    assert out.loss.requires_grad
    assert out.prediction.shape == (bsz, 12, 100)


def test_decoder_with_signal_head_forward_no_labels(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=True)
    bsz = 2
    tgt_ids = torch.randint(1, 100, (bsz, 32))
    signal = torch.randn(bsz, 12, 100)
    out = model(tgt_ids, signal)
    assert out.loss.shape == ()
    assert out.prediction.shape == (bsz, 12, 100)


def test_decoder_with_signal_head_freeze(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=True)
    for p in model.decoder.parameters():
        assert not p.requires_grad
    for p in model.signal_head.parameters():
        assert p.requires_grad


def test_decoder_with_signal_head_generate_signal(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=True)
    model.eval()
    tgt_ids = torch.randint(1, 100, (1, 16))
    signal_shape = (1, 12, 100)
    out = model.generate_signal(tgt_ids, max_new_tokens=10, signal_shape=signal_shape, num_steps=5)
    assert out.shape == signal_shape


def test_decoder_with_signal_head_generate_passthrough(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=False)
    model.eval()
    tgt_ids = torch.randint(1, 100, (1, 16))
    tokens = model.generate(tgt_ids, max_new_tokens=5, return_logits=False)
    assert tokens.shape[1] == 21


def test_decoder_with_signal_head_resize(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=False)
    model.resize_embeddings(150)
    assert model.cfg.vocab_size == 150


def test_cross_attention_uses_full_context(decoder, signal_head):
    model = DecoderWithSignalHead(decoder, signal_head, freeze_decoder=True)
    bsz = 1
    short_ids = torch.randint(1, 100, (bsz, 8))
    long_ids = torch.randint(1, 100, (bsz, 32))
    signal = torch.randn(bsz, 12, 100)
    out_short = model(short_ids, signal)
    out_long = model(long_ids, signal)
    assert not torch.allclose(out_short.loss, out_long.loss, atol=1e-6)


def test_signal_flow_head_backward(signal_head):
    context = torch.randn(2, 20, 64, requires_grad=True)
    signal = torch.randn(2, 12, 100)
    out = signal_head(context, signal)
    out.loss.backward()
    assert context.grad is not None
    for p in signal_head.parameters():
        if p.requires_grad:
            assert p.grad is not None
