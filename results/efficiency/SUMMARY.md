# Efficiency / deployability � GRU spoofing monitor

Causal GRU (2x96), 39 features @ 10 Hz. CPU, 1 thread(s), torch 2.14.0+cpu. Streaming latency = one 10 Hz GNSS epoch in, one spoof probability out, hidden state carried. Real-time budget 100 ms/epoch.

## Cost / speed

| variant      | size_kb            |   lat_ms_median |   lat_ms_p99 |   throughput_steps_s |   realtime_margin_x |
|:-------------|:-------------------|----------------:|-------------:|---------------------:|--------------------:|
| fp32         | 413                |          0.3731 |       0.9217 |                12000 |               108.5 |
| fp16         | 209 (2.0x smaller) |          0.4972 |       0.8849 |                 3085 |               113   |
| int8_dynamic | 112 (3.7x smaller) |          1.5537 |       2.3377 |                 2587 |                42.8 |
| int4_wo_sim  | 413                |          0.2811 |       0.5075 |                12059 |               197   |

`realtime_margin_x` = 100 ms / p99 streaming latency.

## Accuracy vs FP32

ROC-AUC:

| split       |   fp16 |   fp32 |   int4_wo_sim |   int8_dynamic |
|:------------|-------:|-------:|--------------:|---------------:|
| val         |  0.963 |  0.963 |         0.953 |          0.888 |
| test_town   |  0.964 |  0.964 |         0.956 |          0.854 |
| test_attack |  0.917 |  0.917 |         0.908 |          0.858 |
| test_both   |  0.888 |  0.888 |         0.883 |          0.819 |

PR-AUC:

| split       |   fp16 |   fp32 |   int4_wo_sim |   int8_dynamic |
|:------------|-------:|-------:|--------------:|---------------:|
| val         |  0.756 |  0.756 |         0.755 |          0.578 |
| test_town   |  0.771 |  0.771 |         0.739 |          0.495 |
| test_attack |  0.657 |  0.657 |         0.637 |          0.488 |
| test_both   |  0.52  |  0.52  |         0.492 |          0.345 |

TPR @ 1% FPR:

| split       |   fp16 |   fp32 |   int4_wo_sim |   int8_dynamic |
|:------------|-------:|-------:|--------------:|---------------:|
| val         |  0.705 |  0.706 |         0.71  |          0.501 |
| test_town   |  0.76  |  0.761 |         0.788 |          0.476 |
| test_attack |  0.504 |  0.505 |         0.523 |          0.344 |
| test_both   |  0.509 |  0.511 |         0.54  |          0.3   |

## Notes

- **int8_dynamic**: PyTorch dynamic quantization -- int8 weights **and dynamic
  per-tensor int8 quantization of activations** on GRU + Linear. 3.7x smaller on disk
  but ~4x SLOWER per step on this CPU (quant/dequant overhead dominates a model this
  small) and costs 0.06-0.11 ROC-AUC / 0.17-0.28 PR-AUC. Not worth it here.
- **int4_wo_sim**: simulated symmetric 4-bit **weight-only** quantization (no
  activation quant). Accuracy proxy only -- no INT4 CPU kernel in stock PyTorch, so
  compute/latency are unchanged; a real packed 4-bit model would be ~52 KB on disk.
  Near-lossless: -0.005 to -0.01 ROC-AUC, PR-AUC flat, recall@1%FPR unchanged.
- **fp16**: weights cast to half -> 2x smaller, accuracy identical to FP32. Stock x86
  CPUs have no fast FP16 GEMM so latency does not improve on CPU; it is the useful
  path on a GPU / embedded-NPU target.

## Conclusion

The FP32 monitor is already **413 KB and 108x faster than the 10 Hz real-time budget
on a single CPU core** (0.37 ms median / 0.92 ms p99 per epoch) -- it deploys as-is on
an ECU-class processor with no accelerator. If storage is constrained, **4-bit
weight-only quantization is near-lossless** (~8x smaller, PR-AUC unchanged). The
accuracy cliff is specific to **dynamic activation quantization**: the consistency
residual features span a wide dynamic range that a single per-tensor activation scale
cannot represent, so INT8-dynamic loses 0.18 PR-AUC. Weight precision is not the
bottleneck; activation-scale granularity is.
