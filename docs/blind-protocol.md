# Blind-test protocol

The fixture author plants the changes and keeps the answer key. The detector
operator gets the model pair and a SHA-256 digest of the key. Both parties agree
on one run before ranking. The key lists one tensor name per line. The author
can add an unpredictable secret in a `#` comment to keep a short list of
guessable names from being recovered by trying candidate hashes.

From the repository root, use the same Python environment for all three steps.
Set `PYTHONPATH=src` so Python imports the tracked detector from this checkout.

1. **Freeze.** The detector operator runs this before examining the fixture.

   ```sh
   python scripts/blind.py freeze --run RUN
   ```

   `RUN/freeze.json` records HEAD, a SHA-256 for each detector source and the
   scorer, the Python and NumPy versions, and the UTC freeze time. Git must
   report the detector files tracked and clean in both the index and worktree.

2. **Rank.** The author sends only the digest of the exact key-file bytes.
   The operator passes that digest and the checkpoints to the default scanner.

   ```sh
   # Fixture author
   python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('key.txt').read_bytes()).hexdigest())"
   # Detector operator
   python scripts/blind.py rank --run RUN --base BASE --suspect SUSPECT --key-sha256 HEX
   ```

   The operator gives the author `RUN/frozen-ranking.json` before requesting
   the key. It binds the completed `scan/ranking.csv`, the key commitment and
   the exact `freeze.json` bytes to the UTC ranking time. Rank checks the
   detector before and after the scan. An interrupted or rejected scan stays
   in its directory. A fresh attempt requires a new run and a still-blind operator.

3. **Score.** The author releases the committed key. The operator runs

   ```sh
   python scripts/blind.py score --run RUN --key PATH
   ```

   Score checks both records, the ranking and the detector before opening PATH.
   Once the key hash matches, `scripts/score_blind.py` judges the ranking.
   `RUN/result.json` records the frozen commit, ranking and key hashes, scorer
   output lines and exit code. PASS exits 0. FAIL exits 1 and is a recorded result.
   A repeat score or rank refuses. Concurrent commands cannot replace a
   record. If a crash leaves a publication lock, preserve the run as evidence.

Someone with the same account can edit the files or start another run.
A key can leak outside the tool, and a detector can be tuned on earlier
fixtures. A PASS establishes the scorer's top-ranked set matched this key.
The model's safety and the origin of its changes remain open questions.
