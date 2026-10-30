# Repository Guidelines

## Scope

- This repository owns the `seren-memory-sdk` Rust library crate: the client-side memory layer for Seren Desktop and other Seren applications. It provides an offline local cache, sync with the cloud memory service, session bootstrap context, and an MCP client for remember and recall.
- The cloud memory service, its storage, and its API live in `seren-memory`. Desktop integration and its dependency pin live in the consuming application, such as `seren-desktop`. This repository has no binaries, services, or deployment configuration.
- Read `README.md` before changing the public API or the compatibility notes it documents.

## Structure

- `src/lib.rs` is the crate entry point. It declares the public modules and re-exports `SdkError`, `SdkResult`, and `chrono`.
- `src/cache.rs` holds `LocalCache`, the SQLite cache with `sqlite-vec` vector search and FTS5 keyword search. Its schema is created and upgraded in place when the cache opens.
- `src/client.rs` holds `MemoryClient`, the HTTP and MCP client. `src/sync.rs` holds `SyncEngine`. `src/bootstrap.rs` assembles session context. `src/models.rs` and `src/error.rs` hold shared types and errors.
- Tests are inline `#[cfg(test)]` modules in each source file. They use `LocalCache::open_in_memory`, `tempfile` directories, and `wiremock` mock servers.
- There is no generated code, build script, `Makefile`, CI workflow, or Git hook. `Cargo.lock` is committed.

## Validation

- `cargo fmt --all -- --check` checks formatting.
- `cargo clippy --all-targets --locked -- -D warnings` runs lint checks.
- `cargo test --locked` runs the test suite.
- The crate has no Cargo features and is not a workspace, so these commands need no feature or workspace flags. Run all three before handing off a change and report which ran.
- Name tests for the behavior they prove. Test through the public types and mock servers, never against a live Seren endpoint.
- When a consumer needs a change here, validate it from a disposable consumer checkout with a temporary `[patch]` override. Do not commit the override or the lockfile it produces.

## Security and Operations

- `MemoryClient` holds the caller's Seren API key and sends it only as a bearer token to the configured base URL. Never log, print, hardcode, or commit an API key, and never add one to tests or fixtures.
- The local cache stores memory content, embeddings, and sync state in a SQLite file at a path the consumer chooses. Keep user memory data out of the repository and out of test fixtures.
- Keep the bounded connect and request timeouts on every HTTP call, and surface server and MCP tool errors to the caller instead of hiding them.
- Production is the consumer's release. A change reaches users only when a consumer updates its pin, so document public API and compatibility changes in `README.md` in the same change.

## Changes and Commits

- Use Conventional Commits with a single-line subject, such as `fix(cache): delete memories by session`.
- Do not include issue numbers, ticket URLs, signatures, co-author trailers, generated-by notices, or tool attribution.
- Do not record issue or pull request numbers, URLs, or commit hashes as history in code, comments, or test names.
- Never bypass hooks with `--no-verify`.
- Keep changes scoped and preserve unrelated work.

## Git Remotes

- Remote publication is blocked by default and remains owned by the user.
- An agent may publish one ordinary, non-force branch ref only when the user explicitly authorizes that exact push operation in the current turn.
- The exact authorization syntax is a single standalone line:
  `AUTHORIZE_GIT_PUSH_ONCE: <exact direct git push command>`.
  It authorizes only that command, in that working directory, once during the current turn.
- Force pushes, tag pushes, mirror pushes, remote-ref deletions, and multi-ref pushes remain prohibited.
- Local commits are allowed when requested.
