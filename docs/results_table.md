## Experiment Results Table

| config    | qat   |   epochs | learnable   |   max_diff |   mean_diff |   fp32_ms |   dequant_ms |
|:----------|:------|---------:|:------------|-----------:|------------:|----------:|-------------:|
| ptq       | False |        0 | False       |    1.61951 |    0.33419  |   5.99904 |      4.45583 |
| qat_short | True  |        3 | False       |    1.47782 |    0.245477 |   7.21419 |      3.95546 |
| qat_scale | True  |        3 | True        |    1.30308 |    0.248101 |   7.62196 |      6.54249 |