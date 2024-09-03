# Changelog

## Version 0.1.1

This release contains some minor fixes and CI work. No breaking changes have been introduced.

[diff since 0.1.0](https://github.com/QEDjl-project/QEDcore.jl/compare/release-0.1.0...release-0.1.1)

### Fixes

- [#42](https://github.com/QEDjl-project/QEDcore.jl/pull/42): Make `mul` function private since `*` should be used instead
- [#47](https://github.com/QEDjl-project/QEDcore.jl/pull/47): Fix docs building to deploy to gh-pages correctly

### Maintenance

- [#44](https://github.com/QEDjl-project/QEDcore.jl/pull/44): Add docs building job to the CI 
- [#46](https://github.com/QEDjl-project/QEDcore.jl/pull/46): Remove `Suppressor.jl` dependency which is not needed anymore

## Version 0.1.0

**Initial Release**

### New features

- [#6](https://github.com/QEDjl-project/QEDcore.jl/pull/6), [#23](https://github.com/QEDjl-project/QEDcore.jl/pull/23), [#25](https://github.com/QEDjl-project/QEDcore.jl/pull/25), [#29](https://github.com/QEDjl-project/QEDcore.jl/pull/29): Setup basic functionality from QEDbase and QEDprocesses
- [#26](https://github.com/QEDjl-project/QEDcore.jl/pull/26): Move QEDbase concrete particle types here
- [#28](https://github.com/QEDjl-project/QEDcore.jl/pull/28): Move `base_state` implementations from QEDbase
- [#30](https://github.com/QEDjl-project/QEDcore.jl/pull/30): Add integration tests
- [#31](https://github.com/QEDjl-project/QEDcore.jl/pull/31): Add convenient constructors for `PhaseSpacePoint`
- [#32](https://github.com/QEDjl-project/QEDcore.jl/pull/32): Use Reexport.jl to export QEDbase's symbols

### Fixes

- [#27](https://github.com/QEDjl-project/QEDcore.jl/pull/27): Fix unit tests when run by integration tests of QEDbase

### Maintenance

- [#15](https://github.com/QEDjl-project/QEDcore.jl/pull/15): Add Gitlab CI setup
- [#16](https://github.com/QEDjl-project/QEDcore.jl/pull/16): Add automatic formatting setup
