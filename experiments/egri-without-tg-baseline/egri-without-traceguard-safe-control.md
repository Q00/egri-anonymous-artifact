# EGRI without TraceGuard safe-parent control

- Fixtures: 8
- Live model calls: `False`
- Variant: `egri_without_traceguard_safe_parent_control`
- Pass: 8/8
- Unsupported commits: 0/8
- Omitted-fact commits: 0/8

| Fixture | Category | Result safe w/o TG | Unsupported commit |
|---|---|---:|---:|
| `simple-truncation-01` | `simple_truncation` | PASS | False |
| `simple-truncation-02` | `simple_truncation` | PASS | False |
| `distractor-heavy-01` | `distractor_heavy` | PASS | False |
| `distractor-heavy-02` | `distractor_heavy` | PASS | False |
| `cross-chunk-dependency-01` | `cross_chunk_dependency` | PASS | False |
| `cross-chunk-dependency-02` | `cross_chunk_dependency` | PASS | False |
| `omitted-fact-temptation-01` | `omitted_fact_temptation` | PASS | False |
| `chunk-only-citation-trap-01` | `chunk_only_citation_trap` | PASS | False |
