# Engineering Principles

## Primary Objective

Maximize remote competition score.

## Secondary Objective

Reduce latency and resource consumption.

## Non-Goals

- Clever code for its own sake
- Premature optimization
- Complexity without measurable gains

# Core Rules

Every change must document:

- CPU impact
- RAM impact
- Latency impact
- Detection impact
- Complexity impact

# Benchmark Philosophy

Local benchmarks are indicators.

Remote benchmarks are truth.

# Design Philosophy

Prefer:

- Deterministic systems
- Explicit behavior
- Measurable improvements
- Small incremental changes

Avoid:

- Hidden state
- Untested optimizations
- Overfitted models
- Architecture astronautics

# Decision Order

1. Correctness
2. Detection quality
3. Reliability
4. Performance
5. Elegance
