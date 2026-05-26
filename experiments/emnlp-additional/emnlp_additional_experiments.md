# EMNLP Additional Experiments

Private deterministic paper artifact. No live model or network calls were made.

## Real multidataset scale-up

Generated 80 deterministic examples across Qasper, HotpotQA, 2WikiMultihopQA, and ASQA-style families.
This is a larger official-family fixture than the earlier 8-example CI mini-suite, but remains deterministic runtime-control evidence.

## Post-hoc checker vs pre-commit TraceGuard

The post-hoc checker flags unsupported citations at rate 1.0000 but blocks before commit at rate 0.0000.
The pre-commit TraceGuard condition blocks before commit at rate 1.0000 and has unsupported commit rate 0.0000.
This separates detection after generation from admissibility before state mutation.

## Deterministic cost/token/latency overhead

The cost/token/latency table reports normalized units, not provider billing. It makes the tradeoff explicit: extra bounded calls buy evidence bandwidth and pre-commit rejection.

| Condition | Calls | Token units | Latency units | Recall | Unsupported commit |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-shot-context-window | 1 | 1680 | 2.0 | 0.375 | 1.0 |
| flat-topk-retrieval | 1 | 1712 | 2.0 | 0.625 | 1.0 |
| recursive-evidence-cache | 8 | 2784 | 9.6667 | 1.0 | 1.0 |
| recursive-traceguard-dfs | 8 | 2784 | 9.6667 | 1.0 | 0.0 |
