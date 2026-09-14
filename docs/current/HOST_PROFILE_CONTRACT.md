# Host Profile Contract

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v1`

`HostProfile` is a read-only point-in-time observation. It does not contain a Production Target label or a validation decision.

## Fields

| Field | Meaning |
| --- | --- |
| `platform` | Normalized OS platform such as `darwin` or `windows`. |
| `architecture` | Normalized architecture such as `arm64` or `x86_64`. |
| `os_version` | Observed OS release. |
| `os_build` | Observed OS build when available. |
| `hardware_model` | Hardware model when the platform exposes it. |
| `cpu_count` | Observed logical CPU count. |
| `physical_memory_bytes` | Observed physical memory, when available. |
| `unified_memory_bytes` | Observed Unified Memory on Apple Silicon; otherwise `null`. |
| `memory_pressure` | Observed pressure label from the host, not a pass/fail. |
| `swap_used_bytes` / `swap_total_bytes` | Host swap observation when available. |
| `mlx_available` | Whether MLX and mlx-lm can be discovered on the host. |
| `metal_available` | Whether the MLX default device is Metal-backed. |
| `mlx_version` / `mlx_lm_version` | Installed engine versions when available. |
| `captured_at` | UTC capture timestamp. |

The profile intentionally does not include `runtime_mode=production-mac`, `production_eligible`, memory thresholds, or target policy. Benchmark may interpret these facts within its own authority.

## Windows behavior

Windows detection is supported for development. MLX/Metal/Unified Memory fields remain unavailable. The Mock profile is used in tests to avoid treating the host's actual Windows memory as a production observation.

## Volatile metrics

Process footprint and before/after swap measurements belong to `RuntimeMetrics`/`ExecutionTrace`, because they describe one execution rather than the static host profile.
