# Scripts

Utilities in this directory are orchestration code around the core package. Keep profiling, command-line wiring, and reporting here so `cs336_basics/bpe.py` can stay focused on the BPE algorithm.

## Profile BPE Training

Run `train_bpe` with section-level timings:

```bash
uv run python scripts/profile_bpe.py tests/fixtures/corpus.en --vocab-size 500
```

By default, this times the merge-loop helpers and reports the remaining `train_bpe` wall time as `pretokenize_and_setup`:

```text
create_bytepair              0.7046s calls=243
create_merge                 0.5824s calls=243
get_best_pair                0.0557s calls=243
pretokenize_and_setup        0.0287s calls=1
```

Use a different corpus or vocabulary size:

```bash
uv run python scripts/profile_bpe.py data/TinyStoriesV2-GPT4-valid.txt --vocab-size 1000
```

Parallelize pretokenization:

```bash
uv run python scripts/profile_bpe.py data/TinyStoriesV2-GPT4-debug-1GB.txt \
  --vocab-size 10000 \
  --num-processes 4
```

Pass special tokens explicitly:

```bash
uv run python scripts/profile_bpe.py tests/fixtures/corpus.en \
  --vocab-size 500 \
  --special-token '<|endoftext|>'
```

Time a custom set of BPE helper functions:

```bash
uv run python scripts/profile_bpe.py tests/fixtures/corpus.en \
  --vocab-size 500 \
  --section get_best_pair \
  --section create_bytepair \
  --section create_merge
```

Disable special-token splitting:

```bash
uv run python scripts/profile_bpe.py tests/fixtures/corpus.en \
  --vocab-size 500 \
  --no-special-tokens
```

## Add cProfile

Use `cProfile` when section timings identify a hot phase and you want function-level detail:

```bash
uv run python scripts/profile_bpe.py tests/fixtures/corpus.en \
  --vocab-size 500 \
  --cprofile-output /tmp/bpe.prof \
  --cprofile-top 30
```

With `--num-processes > 1`, `cProfile` profiles the parent process. The default `pretokenize_and_setup` residual still includes the wall time spent waiting for worker processes.

Inspect the saved profile later:

```bash
uv run python -m pstats /tmp/bpe.prof
```

Useful `pstats` commands:

```text
sort cumtime
stats 30
```

## Use py-spy

Use `py-spy` when you want a sampling profiler or flame graph without changing the BPE code:

```bash
uv run py-spy record -o /tmp/bpe.svg -- \
  uv run python scripts/profile_bpe.py tests/fixtures/corpus.en --vocab-size 500
```

For a live view:

```bash
uv run py-spy top -- \
  uv run python scripts/profile_bpe.py tests/fixtures/corpus.en --vocab-size 500
```

If `py-spy` is not installed in the environment:

```bash
uv add --dev py-spy
```
