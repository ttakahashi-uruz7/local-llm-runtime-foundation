# AGENTS.md

## Repository purpose

This repository is the independent Local LLM Runtime Foundation shared by Novel Studio, Benchmark Studio, and Learning Studio. It is infrastructure, not a fourth Studio and not a user-facing workbench.

Canonical current specifications live under `docs/current/`. Historical Benchmark documents and PR #10 code are references only; they do not override the Foundation contracts.

## Authority boundary

Foundation owns:

- Engine Adapter interface and capability discovery.
- MLX / `mlx-lm` execution, Mock execution, and the future llama.cpp boundary.
- Host observation: platform, architecture, OS, hardware, memory, Metal/MLX availability, memory pressure, and swap observations.
- Artifact binding validation, model load/unload, generation, streaming, cancellation, lifecycle, health, requested/effective settings, raw metrics, and execution traces.
- The versioned local service and Python client boundary.

Foundation does not own:

- Benchmark Model Registry meaning/history/status, Runtime Optimizer, Quick/Full Optimize, scoring, Quality Benchmark, Blind Review, Safety Gate, Production Runtime Validation, Deployment, or Novel Export.
- Novel Production Config, role assignment, production switch, rollback, manuscript, or UI.
- Learning data, training, SFT, LoRA/QLoRA, preference learning, learned asset promotion, or learned model profiles.

Consumers decide whether Foundation observations satisfy their own policies.

## Safety and data rules

- Do not commit model weights, caches, huge artifacts, secrets, or user-specific absolute model paths.
- Do not make automatic model download part of validation. Model acquisition is a separate workflow.
- Do not add cloud/API inference or automatic cloud fallback.
- Do not write Novel Production settings, Benchmark scores, Benchmark Registry status, Deployment status, or Learning authority from this repository.
- Do not hard-code Qwen or any other model family. A Qwen integration is only an integration test target.
- Do not represent `runtime_mode=production-mac`, `production_eligible`, pass/fail thresholds, or quality judgments in Foundation contracts.

## Windows and Mac

Windows is the development and Mock/HTTP contract environment. Real MLX, Metal, Apple Unified Memory, swap behavior, memory pressure, and throughput require an Apple Silicon Mac. Until that validation is complete, report **REAL MAC PRODUCTION VALIDATION PENDING**.

The MLX adapter is lazy and must remain import-safe on Windows. It must not turn the absence of MLX into a cloud fallback or a fake production result.

## Git

Use Sourcetree Embedded Git only:

`C:\Users\user\AppData\Local\Atlassian\Sourcetree\git_local\cmd\git.exe`

Never switch silently to PATH/system Git. Do not use force push, reset --hard, git clean, history rewrite, or destructive changes to sibling repositories. Do not commit directly to `main`; main merge always requires explicit user approval.

The Benchmark reference PR #10 remains open. This repository must not close, merge, or repurpose that PR.

## Autonomous PR workflow

For ordinary implementation, fix, refactor, migration, and documentation tasks, the standard autonomous scope runs through Pull Request update. Treat the following as pre-approved repository work and do not request an individual user confirmation for them:

- Create or update a work branch.
- Edit files, stage changes, and create commits.
- Push a work branch.
- Create a Pull Request.
- Update an existing Pull Request with follow-up commits.
- Run normal CI and repository validation.

Branch push, Pull Request creation, and Pull Request updates are not an external-publication approval gate, a production deployment, or an irreversible operation. Do not stop with a message such as “GitHubへのPR作成は外部公開操作のため確認が必要です” or any equivalent request for approval. If work is tied to an existing Pull Request, continue updating that Pull Request; do not close it or create a replacement merely to avoid a follow-up update.

The only normal user approval gate is merging to `main`. Never make a direct commit to `main`, merge a Pull Request, or merge to `main` unless the user explicitly instructs it with wording such as “mainにマージして”, “mergeして”, or equivalent. The `main` merge restriction must never be used as a reason to stop commits, pushes, Pull Request creation, or Pull Request updates.

Do not speculate about GitHub permissions, OAuth scopes, or credentials and wait for confirmation. Attempt ordinary push, Pull Request creation, and Pull Request update first. Only an actual authentication, permission, or scope refusal is a blocker; investigate that refusal without force push, history rewrite, credential-setting changes, or scope bypasses.

Choose the work branch and base from repository history, current specifications, existing Pull Requests, existing branches, and task scope whenever those facts make the choice safe. Do not ask the user to resolve routine branch/base ambiguity that repository investigation can resolve.

## PR Completion Gate / Definition of Done

For every ordinary implementation, fix, refactor, migration, or documentation task that changes files, the complete unit of work is:

`branch -> implementation -> validation -> commit -> push -> Pull Request create/update`

The task is **INCOMPLETE** until the Pull Request creation or update has been verified. The following states are also incomplete:

- Changes exist only locally.
- Changes are committed but not pushed.
- A work branch is pushed but no Pull Request exists.
- A branch associated with an existing Pull Request is pushed but the Pull Request update has not been confirmed.

Clean working tree, passing tests, a completed commit, or a completed push alone do not satisfy this gate. A final report may use completion language only after the Pull Request number and URL are confirmed. Before that confirmation, do not report `SUCCESS`, `COMPLETE`, `DONE`, `MERGE READY`, `DEVELOPMENT COMPLETE`, or equivalent Japanese completion wording such as `作業完了` or `完了しました`.

Branch creation, commit, push, Pull Request creation, Pull Request update, and Pull Request body update are standard pre-approved development operations. Do not ask whether it is acceptable to create a Pull Request, push to GitHub, or update a Pull Request, and do not stop immediately before those operations.

The only valid reason to stop before the Pull Request create/update gate is an actual HARD STOP defined by this file, such as an authentication, permission, or OAuth/scope rejection encountered while attempting the operation, repository corruption, an unpreservable user change, or a required destructive conflict. Do not stop based on speculation that permission may be missing or that confirmation may be needed.

For a multi-repository task, every changed repository must independently reach verified Pull Request create/update state before the overall task may be reported as complete. The final report must include, for each repository: branch, base SHA, final local SHA, remote SHA, commit, push, PR number, PR URL, PR updated/created status, Git status, `Force push: NO`, and `Main merge: NO`.

Pull Request completion is separate from merge approval. Pull Request creation and update are autonomous; merging a Pull Request or `main` still requires explicit user approval.

## Codex protocol

All multi-step work is reported as `Step X/N`. If scope expands, update N. Investigate repository state and current specifications before asking routine design questions. Fix ordinary test failures, type errors, missing fixtures, and small contract decisions autonomously. Do not treat the following as a HARD STOP or user confirmation gate: ordinary test failures, implementation bugs, type errors, lint failures, missing fixtures, additional test needs, documentation updates, commit creation, branch push, Pull Request creation, Pull Request updates, lack of separate permission for Pull Request creation, or the absence of a Mac Production Validation environment. Without a Mac, complete all safe Windows development, tests, commits, pushes, and Pull Request updates.

The only HARD STOP conditions are:

- Repository identity is unknown.
- Repository corruption prevents safe work.
- Existing user changes cannot be preserved while proceeding.
- Canonical specifications contain an unresolvable contradiction.
- An actual credential, authentication, or permission refusal prevents the requested Git operation.
- The requested result can be achieved only through a destructive operation.

If none of those conditions applies, complete the original request through Pull Request update.

Before commit:

1. Inspect diff, staged paths, secrets, and existing changes.
2. Run focused/full tests and compile/type checks available in the repository.
3. Run `git diff --check`.
4. Record Base SHA, final SHA, remote SHA, and clean status.

## Mac validation gate

Before calling the Foundation MLX path production validated, run the current Mac runbook, record engine versions/builds, artifact binding, chat-template behavior, load/unload cleanup, cold/warm TTFT, prefill/generation metrics, process footprint, peak memory, memory pressure, swap, cancellation, and failure evidence. No Windows/Mock result can substitute for those observations.

## GitHub Actions Resource Policy

GitHub Actions minutes are a limited resource.

During iterative development, prefer complete local validation from the canonical repository whenever the required checks can be performed locally. Do not intentionally trigger or re-run GitHub Actions merely to test speculative fixes. Investigate a failed remote run first, then fix and validate locally before another remote run whenever local reproduction is possible.

Canonical local validation is:

```powershell
python -m pytest --basetemp=.pytest-temp
```

Run it from `C:\Projects\local-llm-runtime-foundation`. The repository currently has no GitHub Actions workflow and zero recorded Actions runs, so do not add a workflow solely for uniformity or convenience. If a future remote-only or final validation gate is necessary, document why it cannot reasonably be replaced locally and use the minimum required run(s).

Actions cost reduction must never be achieved by silently skipping required quality checks. PR creation and PR updates must not be suppressed to save Actions minutes; Pull Requests remain the formal Sol Review surface. GitHub MCP/API use for repository, PR, review, comment, and CI inspection does not itself consume the constrained GitHub-hosted runner resource. Do not rely on `[skip ci]` as a routine policy or repeatedly re-run Actions without a concrete reason.

## Canonical Repository Policy

- macOS canonical working copy: `/Users/takahashitoru/Projects/local-llm-runtime-foundation`.
- Windows canonical working copy: `C:\Projects\local-llm-runtime-foundation`.
- Codex implementation, file changes, Git operations, tests, builds, launchers, and other repository-targeting operations must use only the path for the executing OS. The Global `~/.codex/AGENTS.md` is authoritative for the complete four-repository canonical path list.
- Do not create or use a new clone, repository directory copy, `git worktree add`, Codex isolated checkout, temporary / recovery / trapfix repository, or alternate checkout. Do not fall back to `/tmp`, `Documents/ChatGPT`, or any other non-canonical path.
- A dirty working tree is not a reason to create another clone, worktree, or copy. Preserve existing changes and separate work with ordinary Git branches in this canonical repository.
- If the canonical path is missing, unavailable, or its state is unknown, do not create a replacement; HARD STOP. Create a repository or worktree only when the user explicitly approves it.
- At task start, verify repository root, origin, branch, HEAD, and working-tree state. Never perform repository work from a non-canonical path.

## Two-Failed-Fix Root-Cause Escalation Policy

Count one failure attempt only when a cause-based fix is applied and verification shows that the same symptom remains. Re-running the same command, retrying a network request, or rerunning external CI does not count. After two attempts for the same symptom, do not apply a third similar local patch; move to Root-Cause Analysis.

In order, fix the reproduction conditions; list prior hypotheses, fixes, and verification results; separate facts from assumptions; list multiple hypotheses; trace upstream through configuration source, authority / source of truth, execution path, call chain, persisted state, runtime state, environment, and external dependency; compare against a known-good state; collect evidence for the root cause; apply the smallest root-cause fix; recheck the original symptom; and run regression checks for previously resolved related behavior.

Before the root cause is proven, do not stack small condition changes, try/except additions, fallbacks, retries, special cases, or relaxed validation. If the evidence cannot establish the root cause, HARD STOP and do not present an unverified hypothesis as fact.

## GitHub MCP mandatory operation path

GitHub固有の参照・操作では、GitHub MCPを正規かつ最優先の経路とする。

- Pull Request、Issue、review、comment、check / CI status、GitHub上のbranch / commit / repository metadata、その他GitHub APIで扱う情報の参照・操作は、利用可能なGitHub MCPを最初に使用する。
- GitHubのWeb UIをbrowser、Computer Use、browser automation、Playwright等で操作してはならない。
- GitHub MCPで実行可能な操作を、Web UI操作で代替してはならない。
- MCPよりWeb UIの方が簡単、MCP操作に失敗した、または権限不足の可能性がある、という理由だけでbrowserへfallbackしてはならない。
- GitHub MCPが利用不能・未接続・必要操作を非対応・実際のauthentication / permission / scope拒否で継続不能な場合は、そのGitHub操作をHARD STOPとして報告する。browserへ自動fallbackしない。
- browser経由のGitHub操作は、HARD STOP報告後にユーザーがその操作について明示的に許可した場合だけ例外的に可能とする。
- GitHub MCPが利用可能か不明な場合は、browserを開く前に利用可能なtool / MCPを確認する。確認せずWeb UIへ進まない。
- repository内の通常のlocal Git操作には、本ファイルで別途定義されたSourcetree Embedded Git規約を適用する。GitHub MCP優先規約を理由に、正規local GitをPATH/system Gitへ切り替えない。
- branch作成、commit、通常push、PR作成・更新、PR comment、CI / check確認等について、既存規約上ユーザー確認不要である場合は、MCP利用を理由に追加確認を要求しない。
- main / PR merge、auto-merge有効化、force-push、branch / PR削除・close、権限変更、その他既存規約で明示承認が必要な操作は、MCPを使用する場合でも同じ承認境界を維持する。

GitHub操作経路の優先順位は次の通りとする。

1. GitHub MCP。
2. repository規約で許可されたSourcetree Embedded Gitによるlocal Git操作。
3. ユーザーが明示的に指定したその他の非browser経路。
4. GitHub Web UI / browser / Computer Useは原則禁止。上記の明示例外時のみ使用可能。
