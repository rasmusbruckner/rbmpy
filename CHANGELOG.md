# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.0.4] — 2026-11-10
### Added
- Linear regression support alongside circular models, allowing estimation in unbounded task spaces.
- New configuration option `min_prob` to adjust numerical safety thresholds for likelihood calculations.
- Expanded unit tests covering linear models, mixtures, and extreme response outliers.

### Changed
- Improved mixture model likelihood computation to treat perseveration with exact mathematical probabilities rather than numerical approximations.
- Updated perseveration scaling so coordinate units are handled consistently across both circular and linear tasks.
- Standardized BIC calculations using the canonical formula across regression models.
- Hardened boundary checks and numerical stability guards during optimization.

## [0.0.3] — 2026-08-15
### Added
- Added `ChangePointTask` for task simulations.
- Added the corresponding `TaskVars` class for task variables.

### Changed
- Updated the perseveration code in `RegressionParent`.
- Updated utilities.

## [0.0.2] — 2025-11-16
### Added
- Added documentation website URL to project metadata.
- Updated project metadata in `pyproject.toml`.

## [0.0.1] — 2025-11-16
### Initial release
- Initial release of `rbmpy`.
