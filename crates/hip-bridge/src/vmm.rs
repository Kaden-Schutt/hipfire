//! Explicit ownership for HIP virtual-memory mappings.
//!
//! `DeviceBuffer` and the existing GPU pool are released with `hipFree`, which
//! is invalid for addresses reserved through the VMM API. `VmmArena` therefore
//! keeps physical handles and mapping ranges separate and requires an explicit
//! [`VmmArena::release`] call.

use crate::{
    DeviceBuffer, HipError, HipMemAccessDesc, HipMemAllocationProp, HipMemGenericAllocationHandle,
    HipResult, HipRuntime, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use std::cell::Cell;
use std::ffi::c_void;

/// Deterministic, test-only fault injection for VMM teardown/access stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmmFaultKind {
    /// Fail the next `hipMemSetAccess` call(s).
    AccessReset,
    /// Fail the next `hipMemUnmap` call(s).
    Unmap,
    /// Fail the next physical-handle `hipMemRelease` call(s).
    Release,
}

thread_local! {
    static FAULT_ACCESS: Cell<u32> = const { Cell::new(0) };
    static FAULT_UNMAP: Cell<u32> = const { Cell::new(0) };
    static FAULT_RELEASE: Cell<u32> = const { Cell::new(0) };
}

/// Queue `count` deterministic failures for `kind`. Test-only; real HIP is
/// unchanged when the counter is zero.
pub fn inject_vmm_fault(kind: VmmFaultKind, count: u32) {
    match kind {
        VmmFaultKind::AccessReset => FAULT_ACCESS.with(|c| c.set(count)),
        VmmFaultKind::Unmap => FAULT_UNMAP.with(|c| c.set(count)),
        VmmFaultKind::Release => FAULT_RELEASE.with(|c| c.set(count)),
    }
}

/// Clear every pending injected VMM fault.
pub fn clear_vmm_faults() {
    FAULT_ACCESS.with(|c| c.set(0));
    FAULT_UNMAP.with(|c| c.set(0));
    FAULT_RELEASE.with(|c| c.set(0));
}

fn take_fault(kind: VmmFaultKind) -> Option<HipError> {
    let cell = match kind {
        VmmFaultKind::AccessReset => &FAULT_ACCESS,
        VmmFaultKind::Unmap => &FAULT_UNMAP,
        VmmFaultKind::Release => &FAULT_RELEASE,
    };
    cell.with(|c| {
        let left = c.get();
        if left == 0 {
            return None;
        }
        c.set(left - 1);
        let label = match kind {
            VmmFaultKind::AccessReset => "access-reset",
            VmmFaultKind::Unmap => "unmap",
            VmmFaultKind::Release => "release",
        };
        Some(HipError::new(
            0x564D_4D46, // 'VMMF'
            &format!("injected VMM {label} failure"),
        ))
    })
}

#[derive(Debug)]
struct VmmSegment {
    offset: usize,
    size: usize,
    handle: Option<HipMemGenericAllocationHandle>,
    mapped: bool,
}

#[must_use = "VMM arenas must be explicitly released with VmmArena::release"]
pub struct VmmArena {
    base: *mut c_void,
    owner_device: i32,
    granularity: usize,
    reserved_bytes: usize,
    mapped_bytes: usize,
    segments: Vec<VmmSegment>,
    access_devices: Vec<i32>,
    releasing: bool,
}

// The HIP process address and allocation handles may move with model state.
// Concurrent mutation is still excluded because VmmArena is not Sync.
unsafe impl Send for VmmArena {}

impl VmmArena {
    pub fn reserve(hip: &HipRuntime, owner_device: i32, requested_bytes: usize) -> HipResult<Self> {
        if requested_bytes == 0 {
            return Err(HipError::new(
                0,
                "VMM reserve size must be greater than zero",
            ));
        }
        let count = hip.device_count()?;
        if owner_device < 0 || owner_device >= count {
            return Err(HipError::new(
                0,
                &format!("VMM owner device {owner_device} is outside available range 0..{count}"),
            ));
        }

        hip.set_device(owner_device)?;
        let prop = HipMemAllocationProp::device_pinned(owner_device);
        let granularity =
            hip.mem_get_allocation_granularity(&prop, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED)?;
        if granularity == 0 {
            return Err(HipError::new(
                0,
                "HIP returned zero VMM allocation granularity",
            ));
        }
        let reserved_bytes = round_up(requested_bytes, granularity)?;
        let base = hip.mem_address_reserve(reserved_bytes, granularity)?;

        Ok(Self {
            base,
            owner_device,
            granularity,
            reserved_bytes,
            mapped_bytes: 0,
            segments: Vec::new(),
            access_devices: vec![owner_device],
            releasing: false,
        })
    }

    pub const fn owner_device(&self) -> i32 {
        self.owner_device
    }

    pub const fn granularity(&self) -> usize {
        self.granularity
    }

    pub const fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    pub const fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    pub fn base_address(&self) -> usize {
        self.base as usize
    }

    pub fn is_released(&self) -> bool {
        self.base.is_null()
    }

    pub fn map_next(
        &mut self,
        hip: &HipRuntime,
        size: usize,
        access_devices: &[i32],
    ) -> HipResult<()> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
        }
        if size == 0 || !size.is_multiple_of(self.granularity) {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM map size {size} must be a non-zero multiple of granularity {}",
                    self.granularity
                ),
            ));
        }
        let next_mapped = self
            .mapped_bytes
            .checked_add(size)
            .ok_or_else(|| HipError::new(0, "VMM mapped byte count overflowed"))?;
        if next_mapped > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM map would exceed reserve: {} + {size} > {}",
                    self.mapped_bytes, self.reserved_bytes
                ),
            ));
        }

        let count = hip.device_count()?;
        let mut devices = Vec::with_capacity(access_devices.len() + 1);
        devices.push(self.owner_device);
        for &device in access_devices {
            if device < 0 || device >= count {
                return Err(HipError::new(
                    0,
                    &format!("VMM access device {device} is outside available range 0..{count}"),
                ));
            }
            if device != self.owner_device && !hip.can_access_peer(device, self.owner_device)? {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM access device {device} cannot access owner device {}",
                        self.owner_device
                    ),
                ));
            }
            if !devices.contains(&device) {
                devices.push(device);
            }
        }
        let mut next_access_devices = self.access_devices.clone();
        for device in devices {
            if !next_access_devices.contains(&device) {
                next_access_devices.push(device);
            }
        }
        let access: Vec<_> = next_access_devices
            .iter()
            .copied()
            .map(HipMemAccessDesc::read_write_device)
            .collect();

        hip.set_device(self.owner_device)?;
        let prop = HipMemAllocationProp::device_pinned(self.owner_device);
        let handle = hip.mem_create(size, &prop)?;
        let address = offset_ptr(self.base, self.mapped_bytes);
        if let Err(err) = unsafe { hip.mem_map(address, size, handle) } {
            return match unsafe { hip.mem_release(handle) } {
                Ok(()) => Err(err),
                Err(cleanup) => {
                    self.segments.push(VmmSegment {
                        offset: self.mapped_bytes,
                        size,
                        handle: Some(handle),
                        mapped: false,
                    });
                    self.releasing = true;
                    Err(combined_cleanup_error(err, cleanup))
                }
            };
        }
        // ROCm 7.2 on gfx1100 accepts 4 KiB allocation granularity but rejects
        // hipMemSetAccess when a later subrange begins at some otherwise-valid
        // 4 KiB offsets (for example base+16 KiB). Reapplying access from the
        // reservation base over the contiguous mapped prefix is accepted and
        // also ensures newly-added peer devices gain access to older segments.
        if let Err(err) = take_fault(VmmFaultKind::AccessReset).map_or_else(
            || unsafe { hip.mem_set_access(self.base, next_mapped, &access) },
            Err,
        ) {
            let err = HipError {
                code: err.code,
                message: format!(
                    "{}; VMM access prefix base=0x{:x} size={} (new segment address=0x{:x} offset={} size={}) granularity={}",
                    err.message,
                    self.base as usize,
                    next_mapped,
                    address as usize,
                    self.mapped_bytes,
                    size,
                    self.granularity,
                ),
            };
            let mut segment = VmmSegment {
                offset: self.mapped_bytes,
                size,
                handle: Some(handle),
                mapped: true,
            };
            let cleanup_error = match unsafe { hip.mem_unmap(address, size) } {
                Ok(()) => {
                    segment.mapped = false;
                    match unsafe { hip.mem_release(handle) } {
                        Ok(()) => {
                            segment.handle = None;
                            None
                        }
                        Err(cleanup) => Some(cleanup),
                    }
                }
                Err(cleanup) => Some(cleanup),
            };
            return match cleanup_error {
                None => Err(err),
                Some(cleanup) => {
                    self.segments.push(segment);
                    self.releasing = true;
                    Err(combined_cleanup_error(err, cleanup))
                }
            };
        }

        // Commit newly requested peer permissions only after the driver has
        // accepted them. A failed expansion must remain retryable with the
        // arena's previous permission set.
        self.access_devices = next_access_devices;
        self.segments.push(VmmSegment {
            offset: self.mapped_bytes,
            size,
            handle: Some(handle),
            mapped: true,
        });
        self.mapped_bytes = next_mapped;
        Ok(())
    }

    /// Return a non-owning buffer view over the reserved virtual address.
    ///
    /// Only the mapped prefix may be accessed. The returned buffer must never
    /// be passed to `HipRuntime::free` or a pool that eventually calls it.
    pub fn buffer(&self, logical_bytes: usize) -> HipResult<DeviceBuffer> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "cannot create a buffer view from a releasing or released VMM arena",
            ));
        }
        if logical_bytes > self.mapped_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM buffer view {logical_bytes} exceeds mapped prefix {}",
                    self.mapped_bytes
                ),
            ));
        }
        Ok(unsafe { DeviceBuffer::from_raw(self.base, logical_bytes) })
    }

    /// Return the unique owning descriptor for a reserved dense tensor.
    ///
    /// # Safety
    ///
    /// The returned buffer's safe byte length is capped at `mapped_bytes()`;
    /// `logical_bytes` only validates the tensor's intended reserved extent.
    /// The caller must register exactly one returned owner for arena teardown.
    pub unsafe fn owner_buffer(&self, logical_bytes: usize) -> HipResult<DeviceBuffer> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "cannot create an owner from a releasing or released VMM arena",
            ));
        }
        if logical_bytes > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM owner view {logical_bytes} exceeds reserve {}",
                    self.reserved_bytes
                ),
            ));
        }
        Ok(DeviceBuffer::from_vmm_owner(
            self.base,
            logical_bytes.min(self.mapped_bytes),
        ))
    }

    /// Unmap every segment, release every physical handle, then free the VA.
    /// Cleanup continues after an individual failure and returns the first one.
    pub fn release(&mut self, hip: &HipRuntime) -> HipResult<()> {
        if self.is_released() {
            return Ok(());
        }
        self.releasing = true;
        for &device in &self.access_devices {
            hip.set_device(device)?;
            hip.device_synchronize()?;
        }
        hip.set_device(self.owner_device)?;
        let base = self.base;
        let mut first_error = cleanup_segments(
            &mut self.segments,
            |offset, size| {
                if let Some(err) = take_fault(VmmFaultKind::Unmap) {
                    return Err(err);
                }
                unsafe { hip.mem_unmap(offset_ptr(base, offset), size) }
            },
            |handle| {
                if let Some(err) = take_fault(VmmFaultKind::Release) {
                    return Err(err);
                }
                unsafe { hip.mem_release(handle) }
            },
        )
        .err();

        if self.segments.is_empty() {
            match unsafe { hip.mem_address_free(self.base, self.reserved_bytes) } {
                Ok(()) => {
                    self.base = std::ptr::null_mut();
                    self.reserved_bytes = 0;
                    self.mapped_bytes = 0;
                    self.access_devices.clear();
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

fn offset_ptr(base: *mut c_void, offset: usize) -> *mut c_void {
    unsafe { (base as *mut u8).add(offset) as *mut c_void }
}

fn round_up(value: usize, alignment: usize) -> HipResult<usize> {
    if alignment == 0 {
        return Err(HipError::new(0, "VMM alignment must be greater than zero"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| HipError::new(0, "VMM reserve size overflowed during alignment"))
    }
}

fn combined_cleanup_error(operation: HipError, cleanup: HipError) -> HipError {
    HipError::new(
        0,
        &format!("{operation}; cleanup also failed: {cleanup}; VMM arena retained for retry"),
    )
}

fn cleanup_segments(
    segments: &mut Vec<VmmSegment>,
    mut unmap: impl FnMut(usize, usize) -> HipResult<()>,
    mut release: impl FnMut(HipMemGenericAllocationHandle) -> HipResult<()>,
) -> HipResult<()> {
    let mut first_error = None;
    for segment in segments.iter_mut().rev() {
        if segment.mapped {
            match unmap(segment.offset, segment.size) {
                Ok(()) => segment.mapped = false,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                    continue;
                }
            }
        }
        if let Some(handle) = segment.handle {
            match release(handle) {
                Ok(()) => segment.handle = None,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
    }
    segments.retain(|segment| segment.mapped || segment.handle.is_some());
    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(mapped: bool) -> VmmSegment {
        VmmSegment {
            offset: 4096,
            size: 4096,
            handle: Some(1usize as HipMemGenericAllocationHandle),
            mapped,
        }
    }

    #[test]
    fn failed_unmap_keeps_mapping_and_handle_for_retry() {
        let mut segments = vec![segment(true)];
        let mut releases = 0;
        let err = cleanup_segments(
            &mut segments,
            |_, _| Err(HipError::new(1, "injected unmap failure")),
            |_| {
                releases += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("injected unmap failure"));
        assert_eq!(releases, 0, "a still-mapped handle must not be released");
        assert!(segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(&mut segments, |_, _| Ok(()), |_| Ok(())).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn failed_handle_release_keeps_handle_for_retry() {
        let mut segments = vec![segment(false)];
        cleanup_segments(
            &mut segments,
            |_, _| panic!("unmap must not run for an unmapped segment"),
            |_| Err(HipError::new(2, "injected handle failure")),
        )
        .unwrap_err();
        assert!(!segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(&mut segments, |_, _| Ok(()), |_| Ok(())).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn inject_vmm_fault_counters_are_consumed_once_each() {
        clear_vmm_faults();
        inject_vmm_fault(VmmFaultKind::Unmap, 2);
        inject_vmm_fault(VmmFaultKind::Release, 1);
        inject_vmm_fault(VmmFaultKind::AccessReset, 1);

        let u1 = take_fault(VmmFaultKind::Unmap).unwrap();
        let u2 = take_fault(VmmFaultKind::Unmap).unwrap();
        assert!(take_fault(VmmFaultKind::Unmap).is_none());
        assert!(u1.to_string().contains("unmap"));
        assert!(u2.to_string().contains("unmap"));

        let r = take_fault(VmmFaultKind::Release).unwrap();
        assert!(r.to_string().contains("release"));
        assert!(take_fault(VmmFaultKind::Release).is_none());

        let a = take_fault(VmmFaultKind::AccessReset).unwrap();
        assert!(a.to_string().contains("access-reset"));
        assert!(take_fault(VmmFaultKind::AccessReset).is_none());
        clear_vmm_faults();
    }

    #[test]
    fn clear_vmm_faults_drops_pending_injections() {
        inject_vmm_fault(VmmFaultKind::Unmap, 5);
        inject_vmm_fault(VmmFaultKind::Release, 5);
        inject_vmm_fault(VmmFaultKind::AccessReset, 5);
        clear_vmm_faults();
        assert!(take_fault(VmmFaultKind::Unmap).is_none());
        assert!(take_fault(VmmFaultKind::Release).is_none());
        assert!(take_fault(VmmFaultKind::AccessReset).is_none());
    }
}
