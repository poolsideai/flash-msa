# Flash-MSA
Flash-MSA is written in CuTeDSL for Hopper and Blackwell GPUs (eg H100, B200) on CUDA 13.

These kernels implement training for the MiniMax Sparse Attention paper:
https://arxiv.org/abs/2606.13392

Briefly, MSA is a style of sparse attention fitted to GQA that uses a small proxy attention layer to select blocks of keys to provide to the main attention layer. This offers a massive speedup to inference by slashing the memory-bandwidth bottleneck of loading the full KV cache from HBM.

The proxy heads are trained via a KL-divergence loss between the main attention layer's attention scores over the sparsely selected blocks. The proxy heads are assigned groups of main attention heads to select keys for & average scores over for KL-teaching.

This library also includes MSA warmup kernels, which run the main attention densely and train the proxy attention on the full sequence.

More information is included in the [blog post](https://nanduruganesh.github.io/flash-msa).

# Installation

The sparse training path requires FA4 block-sparse attention at commit `6a94f8b906cf5ab944385d64707f9387f3dd6be9`, which adds compact block-index tensors. Dense warmup also supports FA3. Install the pinned FA4 dependency with:

```
uv pip install 'flash-msa[fa4]'
```

If you must build from source, set `MAX_JOBS=<max jobs>` to avoid exhausting host memory.

You will also need Python headers, e.g. `apt-get install python3.12-dev`, for whichever python version you are using.

```
uv pip install flash-msa
```

From source:

```
python setup.py install
```
or
```
uv pip install -e . --no-build-isolation
```
# Usage
```
from flash_msa import flash_msa_func
attn_out, kl_loss = flash_msa_func(Q_proxy, K_proxy, Q, K, V, top_k, head_dim ** -0.5)
```
For packed-document attention, pass FA4-style cumulative offsets over flattened
``B * S`` tokens. The offsets must include every batch-row boundary.

```
attn_out, kl_loss = flash_msa_func(
    Q_proxy,
    K_proxy,
    Q,
    K,
    V,
    top_k,
    head_dim ** -0.5,
    cu_seqlens=cu_seqlens,
)
```

or
```
from flash_msa import flash_msa_warmup_func
attn_out, kl_loss = flash_msa_warmup_func(Q_proxy, K_proxy, Q, K, V, top_k, head_dim ** -0.5)
```

Note that kl_loss in the forward is just a torch.zeros placeholder, but after adding it to the main model loss, calling backward() will activate the on-the-fly gradient calcs equivalent to the actual proxy KL loss signal.

To log the actual KL without materializing attention probabilities, pass a scalar FP32
CUDA buffer. The fused backward updates it with the unweighted KL value:

```
kl_metric = torch.zeros((), device=Q.device, dtype=torch.float32)
attn_out, kl_loss = flash_msa_func(
    Q_proxy,
    K_proxy,
    Q,
    K,
    V,
    top_k,
    head_dim ** -0.5,
    kl_metric=kl_metric,
)
loss = model_loss + kl_weight * kl_loss
loss.backward()
```

Frameworks that apply the Indexer projection VJP immediately, or that replay a
full activation checkpoint, can use the decomposed API:

```
from flash_msa import (
    prepare_sparse_attention,
    sparse_main_attention,
    sparse_proxy_vjp,
)

metadata = prepare_sparse_attention(
    Q_proxy, K_proxy, Q, K, V, top_k, head_dim ** -0.5, document_ids
)
attn_out, main_lse = sparse_main_attention(
    Q, K, V, metadata, head_dim ** -0.5
)
dQ_proxy, dK_proxy = sparse_proxy_vjp(
    Q_proxy,
    K_proxy,
    Q,
    K,
    main_lse,
    metadata,
    head_dim ** -0.5,
    kl_metric=kl_metric,
)
```

Apply `dQ_proxy` and `dK_proxy` to the Indexer projection while its short-lived
autograd graph is still available. The ordinary backward of
`sparse_main_attention` computes only main Q/K/V gradients. `metadata` holds the
compact token selections and forward/backward schedules, so a checkpoint replay
can reuse them without rerunning selection or rebuilding either schedule. Dense
warmup provides the same split through `dense_main_attention` and
`dense_proxy_vjp`.

# Caveats

1. Flash-MSA only supports headdims 128, block size 128.
2. Flash-MSA does not return a materialized KL tensor. It can optionally accumulate the scalar KL during backward.
3. No support for quantized training (fp8, nvfp4, mxfp4).
4. No support for attn temps / oai-style softmax bias.
5. The proxy-head count must be at least and divisible by the Main KV-head count.

These are not ridiculous to implement though so if there is demand or if someone makes a PR, I will update the repo to include these features.

# Testing

Test sparse MSA correctness against an eager implementation of MSA: `python tests/test_eager_match.py [args]`

Test warmup MSA correctness against an eager implementation of MSA: `python tests/test_warmup_eager_match.py [args]`

# Training

An MSA training example is implemented in this [Megatron-LM fork](https://github.com/nanduruganesh/Megatron-LM). 

Notably, you must add the kl_loss returned by MSA kernels to the model's main CE loss before backward to train the proxy attention. The kl_loss is a torch.zeros placeholder and calculated on-the-fly in the backward, so logging that placeholder will not reflect how proxy training is actually going. Pass kl_metric when the actual scalar KL is needed.

In general if you are going to train with this it is highly recommended to follow tips from [the paper](https://arxiv.org/abs/2606.13392), use MSA warmup before turning on MSA sparse training, and replicate any transformations to the main attention queries and keys (RoPE, QK norm, QK clip, etc) to the proxy queries and keys to improve proxy convergence.

# Inference
See MiniMax's [official repo](https://github.com/MiniMax-AI/MSA) for MSA inference kernels.
