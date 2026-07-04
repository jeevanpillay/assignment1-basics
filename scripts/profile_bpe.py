from __future__ import annotations

import argparse
import cProfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import pstats

from cs336_basics.bpe import train_bpe
from cs336_basics.profiling import SectionProfiler

DEFAULT_SECTIONS = ("create_bytepair", "get_best_pair", "create_merge")


@contextmanager
def profile_train_bpe_sections(section_names: tuple[str, ...], profiler: SectionProfiler) -> Iterator[None]:
    train_bpe_globals = train_bpe.__globals__
    originals: dict[str, Callable[..., Any]] = {}

    for section_name in section_names:
        original = train_bpe_globals.get(section_name)
        if not callable(original):
            raise ValueError(f"`{section_name}` is not a callable used by train_bpe.")

        originals[section_name] = original

        @wraps(original)
        def wrapped(*args: Any, __name: str = section_name, __original: Callable[..., Any] = original, **kwargs: Any):
            with profiler.section(__name):
                return __original(*args, **kwargs)

        train_bpe_globals[section_name] = wrapped

    try:
        yield
    finally:
        train_bpe_globals.update(originals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile selected helper functions while running train_bpe normally.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--special-token", dest="special_tokens", action="append")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        help=(
            "Helper function to time. Can be passed multiple times. "
            "Defaults to create_bytepair, get_best_pair, and create_merge."
        ),
    )
    parser.add_argument("--cprofile-output", type=Path)
    parser.add_argument("--cprofile-top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_special_tokens:
        special_tokens = []
    else:
        special_tokens = args.special_tokens or ["<|endoftext|>"]

    sections = tuple(args.sections or DEFAULT_SECTIONS)
    section_profiler = SectionProfiler()

    with profile_train_bpe_sections(sections, section_profiler):
        start_time = time.perf_counter()
        if args.cprofile_output is None:
            vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens, args.num_processes)
        else:
            profile = cProfile.Profile()
            profile.enable()
            vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens, args.num_processes)
            profile.disable()
            profile.dump_stats(args.cprofile_output)

            stats = pstats.Stats(profile)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(args.cprofile_top)
        total_time = time.perf_counter() - start_time

    if args.sections is None:
        profiled_time = sum(section_profiler.times.values())
        section_profiler.times["pretokenize_and_setup"] += max(0.0, total_time - profiled_time)
        section_profiler.calls["pretokenize_and_setup"] += 1

    section_profiler.report()
    print(f"\nvocab_size={len(vocab)} merges={len(merges)}")


if __name__ == "__main__":
    main()
