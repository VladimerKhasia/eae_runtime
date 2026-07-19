# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-19

### Changed
- Updated README install instructions to show `pip install eae-runtime` as the
  primary install method, with the editable/dev install moved to its own
  Development section.

## [0.1.0] - 2026-07-19

### Added
- Initial public release of the EAE runtime: block decomposer, forward
  executor, boundary state store, reverse schedulers (sequential, async,
  pipeline, distributed), reconstruction engine, and adjoint pipeline.
- Built-in passes: `ClipPass`, `FP16Pass`, `Int8QuantizationPass` (`FP8Pass`),
  `SyntheticGradientPass`, `RegularizationPass`, `GaussianNoisePass`,
  `LoggingPass`.
- Event bus with structured events and optional JSON logging.
- Profiler for per-stage timing.
- Test suite covering gradient equivalence against `model.backward()`,
  memory management, boundary store, and a real two-process distributed
  correctness check.

[Unreleased]: https://github.com/VladimerKhasia/eae_runtime/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/VladimerKhasia/eae_runtime/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/VladimerKhasia/eae_runtime/releases/tag/v0.1.0