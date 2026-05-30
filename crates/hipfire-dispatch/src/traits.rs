// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::types::DispatchError;

pub trait KernelFamily: Send + Sync {
    fn name(&self) -> &'static str;
}
