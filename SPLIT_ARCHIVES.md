# Split Anonymous Artifact Archives

Upload both archives for review:

- `egri-anonymous-software.zip`: code, replay scripts, table-generation scripts, tests, examples, README, reproducibility notes, and license.
- `egri-anonymous-data.zip`: fixtures, deterministic manifests, replay artifacts, checked-in benchmark outputs, and result metadata.

Both archives expand into the same top-level directory, `egri-anonymous-artifact/`.
Both archives include `README.md` and `LICENSE` so the review upload exposes
artifact terms and reproduction instructions even when software and data are
inspected separately.
To reconstruct the full reproducibility artifact, unzip both archives in the same
parent directory:

```bash
unzip egri-anonymous-software.zip
unzip egri-anonymous-data.zip
cd egri-anonymous-artifact
```

The required deterministic validation path uses the reconstructed tree and does
not require live model credentials or network calls after dependencies are
installed. Optional live-provider reruns are documented in `REPRODUCIBILITY.md`
and remain optional for anonymous review.
