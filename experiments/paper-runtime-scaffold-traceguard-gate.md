# Paper runtime-scaffold TraceGuard Gate

This artifact validates the exact persisted `runtime-scaffold` paper run with
the local deterministic TraceGuard validator.

## Source Run

- Demo artifact: `experiments/paper-key-sections-runtime-scaffold-demo.json`
- Command: `uv run --extra dev runtime-scaffold experiments/paper-runtime-scaffold-key-sections-target.txt --cwd <ARTIFACT_ROOT> --debug`
- Hermes sub-calls: `5`
- RLM tree depth: `1`

## Gate Flow

```text
persisted runtime-scaffold paper run
  |
  | child_result ids + parent accepted claims
  v
normalize claims into TraceGuard fact/evidence handles
  |
  +-- safe parent synthesis
  |     -> TraceGuard ACCEPT
  |
  +-- same parent + MEMORY-ANSWER claim
        -> TraceGuard REJECT
```

## Results

| Case | Accepted | Unsupported rate | Rejection reasons |
| --- | ---: | ---: | --- |
| exact runtime-scaffold parent | true | 0.0000 | none |
| parent + unsafe memory answer | false | 0.0667 | unsupported_fact_id |

## Interpretation

The exact `runtime-scaffold` paper run produces parent claims that can be
normalized into TraceGuard evidence handles and accepted. When the same
parent synthesis is contaminated with an unsupported memory-answer fact,
TraceGuard rejects it as `unsupported_fact_id` because that fact is not
present in the fresh child evidence manifest.

Scope note: this is an automatic post-run gate over the persisted `runtime-scaffold`
run. It proves the end-to-end compatibility of this run with TraceGuard,
but the upstream dependency `runtime-scaffold` command did not invoke
TraceGuard internally when this run was recorded. The project-local
`runtime-scaffold` wrapper now install an in-process gate.
