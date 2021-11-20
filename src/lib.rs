//! Alyx
//!
//! Sparse matrix library.
//!
//! # Sparse matrix formats
//!
//! - [CooMat]: Coordinate sparse matrix format

#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(missing_docs)]

mod coo;
pub use coo::CooMat;
