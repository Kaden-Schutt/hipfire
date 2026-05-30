// SPDX-License-Identifier: MIT OR Apache-2.0

pub trait KernelFamily: Send + Sync {
    fn name(&self) -> &'static str;
}
