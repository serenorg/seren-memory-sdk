// ABOUTME: SDK error types for cache, sync, and client operations.
// ABOUTME: Wraps rusqlite, reqwest, and serialization errors.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SdkError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("server returned {status}: {body}")]
    ServerError { status: u16, body: String },

    #[error("unauthorized")]
    Unauthorized,

    #[error("{0}")]
    Other(String),
}

pub type SdkResult<T> = Result<T, SdkError>;
