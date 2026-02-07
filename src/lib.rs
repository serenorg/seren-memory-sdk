// ABOUTME: Seren Memory SDK for local cache, sync, and session bootstrap.
// ABOUTME: Used by seren-desktop to provide offline-capable memory access.

pub mod bootstrap;
pub mod cache;
pub mod client;
pub mod error;
pub mod models;
pub mod sync;

pub use error::{SdkError, SdkResult};
