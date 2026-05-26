# EGRI without TraceGuard deterministic baseline

- Fixtures: 8
- Live model calls: `False`
- Variant: `egri_without_traceguard`
- Pass: 0/8
- Unsupported commits: 8/8
- TraceGuard would reject committed parent: 8/8
- Omitted-fact commits: 7/8
- Chunk-only-without-fact-id failures: 1/8

| Fixture | Category | Result w/o TG | Failure type | Omitted fact committed |
|---|---|---:|---|---|
| `simple-truncation-01` | `simple_truncation` | FAIL | `unsupported_omitted_fact` | LP-01-005 |
| `simple-truncation-02` | `simple_truncation` | FAIL | `unsupported_omitted_fact` | LP-02-005 |
| `distractor-heavy-01` | `distractor_heavy` | FAIL | `unsupported_omitted_fact` | LP-03-005 |
| `distractor-heavy-02` | `distractor_heavy` | FAIL | `unsupported_omitted_fact` | LP-04-005 |
| `cross-chunk-dependency-01` | `cross_chunk_dependency` | FAIL | `unsupported_omitted_fact` | LP-05-005 |
| `cross-chunk-dependency-02` | `cross_chunk_dependency` | FAIL | `unsupported_omitted_fact` | LP-06-005 |
| `omitted-fact-temptation-01` | `omitted_fact_temptation` | FAIL | `unsupported_omitted_fact` | LP-07-005 |
| `chunk-only-citation-trap-01` | `chunk_only_citation_trap` | FAIL | `chunk_only_without_fact_id` | - |
