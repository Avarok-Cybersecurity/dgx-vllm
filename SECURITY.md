# Security Policy

## Reporting a vulnerability

Please report security issues privately to **contact@avarok.net** rather than
opening a public GitHub issue. A maintainer will follow up to triage and
coordinate a fix.

If you don't get a response within a few business days, feel free to ping a
maintainer directly on github (mention is fine, no details in the public
thread please).

## Supported versions

Only the most recent tagged image is supported for security updates.

| Version    | Supported          |
|------------|--------------------|
| latest (`v22`) | :white_check_mark: |
| < v22      | :x:                |

If you're running an older tag and can't upgrade, mention that in your report
and we'll do our best to advise on mitigations.

## Scope

In scope:

- The Docker image published as `avarok/vllm-dgx-spark` / `avarok/dgx-vllm-nvfp4-kernel`
- The build scripts in this repo (Dockerfile, `*.sh`, `fix_*.py`)
- The custom CUTLASS NVFP4 kernels in `cutlass_nvfp4/` and `sparse_fp4_kernel/`

Out of scope:

- Upstream vLLM, FlashInfer, CUTLASS, or PyTorch issues — please report those
  to the respective projects. We'll happily forward if you're unsure where it
  belongs.
- Issues that require physical access to the host

## Disclosure

We'd appreciate a reasonable embargo window (~90 days) for any kernel-level
findings — these can take real work to patch without breaking the
sm_121a path. For pure dependency-bump issues, just open a PR.
