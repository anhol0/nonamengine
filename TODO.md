# To be done

- Benchmark and optimize CPU matrix multiplication and transpose allocations.
- Add inference mode without training caches.
- Declare CLI11 explicitly in CMake.
- Extend coverage to OS-level read/seek/write failures and unreachable defensive
  branches; current implementation-file line coverage is 98.59% for tensors,
  94.21% for layers/loss, 94.05% for model persistence, and 100% for utilities.
- Cover the full CLI training/export workflow. Training itself has a fast,
  deterministic integration test; the CLI demonstration still hardcodes 20,000
  iterations and its training path is not run by the regular suite.
- Design optimizers and shuffled minibatch training with the user.
- Design a compute-shader GPU backend with the user.

# In progress

None.

# Done

- Split verified changes into focused, signed local commits; keep AGENTS.md
  excluded and outside the index.
- Document authorization for small, focused, signed local commits in AGENTS.md.
- Implement tensor-based CPU neural network execution.
- Implement JSON metadata and binary float32 model export/loading.
- Document project constraints and task tracking in AGENTS.md.
- Add 17 dependency-free CTest groups covering tensors, Dense/ReLU/Tanh, MSE,
  numerical gradient checks, deterministic training, utilities, model loading
  and export, malformed files, failed-load state preservation, and CLI loading.
- Fix empty-dimension broadcasting, missing Dense cache checks, invalid Dense
  dimensions, gradient shape checks, empty MSE inputs, and MSE count narrowing.
- Reject trailing garbage in JSON model files and propagate CLI load/export errors.
- Verify all 17 tests in Debug, Release, and AddressSanitizer/UBSan builds.
  LeakSanitizer is disabled because this environment uses ptrace.

Run the suite with `cmake -S . -B build`, `cmake --build build`, then
`ctest --test-dir build --output-on-failure`.
