# llama.cpp Runtime Design

- Status: **Canonical / Current**
- Engine wire identifier: `llama.cpp`
- Artifact format: `gguf`
- Binding: optional `llama-cpp-python` package and its bundled native llama.cpp library
- Process model: same process as the Foundation Core

## Binding evaluation

| Candidate | Decision | Reason |
| --- | --- | --- |
| `llama-cpp-python` high-level API | **Selected** | Provides GGUF loading, metadata access, streaming chat/completion, sampler options, runtime context parameters, and access to native llama.cpp build information through one maintained binding. Its LoRA constructor fields leave a future integration point. |
| `llama-cli` subprocess | Not selected | Would add a second process/protocol and make token-level settings/effective context evidence and chat-template introspection less direct. This Foundation adapter needs native metadata and runtime state, not only text output. |
| A Foundation-owned ctypes/CFFI wrapper | Not selected | Would duplicate upstream ABI declarations, native lifetime rules, sampler setup, and Metal build handling while adding no required capability over the existing binding. |

The binding is an optional dependency (`llama-cpp-python` 0.3.x); importing
Foundation does not import or require it. The optional `llama-cpp` extra is the
explicit install path. The adapter never installs packages or acquires models.
On Apple Silicon, a Metal-enabled native build must be installed intentionally
using the upstream build configuration. Host Metal and native build Metal are
reported independently.

## Native identity and capability

Engine Build Identity is not the Python package version alone. It includes the
binding distribution version, the native `llama_print_system_info()` output,
the resolved native shared-library location and SHA-256 when those are
observable, and the observed native acceleration markers. Any unavailable
component is explicitly `unknown`; it is not guessed from the host or package
version. A generic GPU-offload flag is not proof of Metal support. Metal build
capability is supported only when the loaded native build reports a concrete
Metal marker; actual Metal execution is recorded separately after a request.

The binding's C API exposes effective context, batch, and micro-batch sizes
after context creation. The adapter records those values. Thread count and
KV/offload requests are recorded as effective only when the loaded binding
exposes an observation; otherwise capability/evidence says the requested value
was passed but not independently observed.

## Artifact, metadata, and chat template

Only consumer-supplied local GGUF files are accepted. Foundation validates the
GGUF container and caller claims before adapter load, binds the full-file
identity, then the adapter pins/revalidates that same identity during native
load. No extension-only check or automatic download is used. GGML is excluded.

Chat requests require an observed GGUF chat template and a handler selected
from that metadata. The adapter explicitly selects the metadata's default
template and constructs the binding's `Jinja2ChatFormatter`; it does not call
the binding's `_chat_handlers` completion wrapper or accept llama.cpp's
implicit llama-2 fallback. It tokenizes the formatter output once and passes
those exact native token IDs to completion, so budget checks and generation
use the same prompt without a second tokenizer path. If the binding cannot
expose or render the selected template, chat generation fails closed with a
template/capability error.

Thinking intent is passed only through template variables the selected
metadata template actually references. `thinking_supported`, effort support,
and budget support are artifact-applicable capability observations. A requested
intent that the template cannot express is rejected before generation.
The exception is a resolved `OFF` (no effort, no budget) on a template whose
variables were inspected and include no thinking control: it renders without
template thinking kwargs and records a `thinking_intent.template_control`
resolution with status `observed`. A template whose variables cannot be
inspected still fails closed.
`llama-cpp-python` LoRA load parameters are a future extension point; this
adapter does not claim direct Base+Adapter support in this phase.

For v3, omitted `top_p`, `top_k`, and repetition penalty resolve from the
loaded binding's inspected `create_completion` defaults and are passed
explicitly. An omitted seed resolves to a Foundation-generated per-request
seed, which is included in execution evidence. The binding does not expose a
request-scoped repetition window, so a non-null `repetition_window` is rejected
instead of being sent as an unsupported argument. The loaded native context is
also checked before inference, even when the caller did not set
`max_context_tokens`; llama.cpp's internal max-token truncation is not treated
as a successful execution.

## Streaming, cancellation, timeout, and failures

Streaming is emitted from the binding's native incremental completion iterator.
Foundation checks cancellation and deadlines between iterator steps and closes
the iterator on termination. This is **cooperative cancellation**: a native
decode step already in progress can delay the response until it returns.
Timeout is cooperative for the same reason. Neither guarantee is hard
termination.

The selected process model is same-process. Python exceptions and native errors
that return through the binding are normalized to Foundation errors. A native
segfault, abort, or process kill can terminate the Foundation service and is
not guaranteed to become an `engine_runtime_error`. Crash isolation would
require a worker-process contract and is not claimed here.

## TEST and release boundary

Unit/contract tests use fake binding objects and synthetic container fixtures;
they do not count as GGUF model integration. A GGUF CPU or Metal integration
Gate requires an explicitly provided TEST model with recorded source/license
and expected SHA-256. No model is downloaded by tests or committed. Metal PASS
requires both a Metal-capable loaded native build and observed Metal execution.

## Upstream references

- [llama-cpp-python high-level API](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py)
- [llama-cpp-python native bindings](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_cpp.py)
- [llama-cpp-python chat template behavior](https://github.com/abetlen/llama-cpp-python#chat-completion)
- [llama-cpp-python macOS Metal build guidance](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md)
- [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases)
