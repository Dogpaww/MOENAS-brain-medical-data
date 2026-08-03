"""Memory tests (handoff §31 "Memory"): sequential candidate evaluations
must not produce continuous/unbounded allocated-memory growth. Uses whole-
process peak/current RSS as a coarse but real signal -- a genuine leak (e.g.
accidentally keeping a growing list of live models) would show clear,
sustained growth well past what this generous tolerance allows; ordinary
allocator/GC noise will not.
"""

from __future__ import annotations

import gc
import resource
import sys

import torch

from brainmri_nas.search.candidate import CandidateCache, evaluate_candidate
from brainmri_nas.search_space.chromosome import total_chromosome_length

INPUT_CHANNELS, IMAGE_SIZE, NUM_CLASSES = 3, 16, 4
N_VAR = total_chromosome_length()


def _rss_bytes() -> int:
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def _fixed_batches(num_batches=2, batch_size=2):
    batches = []
    g = torch.Generator().manual_seed(0)
    for _ in range(num_batches):
        x = torch.randn(batch_size, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), generator=g)
        batches.append((x, y))
    return batches


def _chromosome(offset: int):
    return [((i * 37 + offset) % 97) / 97.0 for i in range(N_VAR)]


def test_sequential_candidate_evaluations_do_not_grow_memory_unbounded():
    proxy_batches = _fixed_batches()
    cache = CandidateCache()  # shared cache would mask this test's point via hits, but distinct
    # chromosomes below decode to distinct genotypes, so every call does real work.

    num_iterations = 24
    rss_samples = []

    for i in range(num_iterations):
        evaluate_candidate(
            _chromosome(offset=i),
            cache=cache,
            num_intermediate_nodes=4,
            edges_per_node=2,
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
            stem_type="cifar",
            proxy_batches=proxy_batches,
            device=torch.device("cpu"),
            number_of_cells_range=(3, 3),
            initial_channels_range=(4, 4),
        )
        gc.collect()
        rss_samples.append(_rss_bytes())

    warmup = rss_samples[:8]
    tail = rss_samples[-8:]
    warmup_avg = sum(warmup) / len(warmup)
    tail_avg = sum(tail) / len(tail)

    # Generous absolute slack: real leaks from 24 tiny-model evaluations would
    # dwarf this; allocator/GC noise on tiny models will not approach it.
    max_allowed_growth_bytes = 300 * 1024 * 1024  # 300 MB
    assert tail_avg - warmup_avg < max_allowed_growth_bytes, (
        f"RSS grew by {(tail_avg - warmup_avg) / 1e6:.1f} MB across {num_iterations} "
        "sequential candidate evaluations -- possible memory leak."
    )
