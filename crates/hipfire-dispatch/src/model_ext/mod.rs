pub mod deepseek4;

// New canonical locations:
#[cfg(feature = "deltanet")]
pub use crate::ops::delta_net::DeltaNetOps;
#[cfg(any())]
pub use crate::ops::mla::MlaOps;

// Deprecated aliases — remove once all callers migrate:
#[cfg(any())]
pub use MlaOps as Deepseek4ModelExt;
