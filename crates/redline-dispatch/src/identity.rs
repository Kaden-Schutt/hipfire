use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

use crate::KernelArg;

pub const DEVICE_POINTER_BYTES: u32 = 8;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn digest(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..8] {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str("…")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernargField {
    offset: u32,
    size: u32,
    alignment: u32,
}

impl KernargField {
    pub fn new(offset: u32, size: u32, alignment: u32) -> Result<Self, KernargAbiError> {
        if size == 0 {
            return Err(KernargAbiError::EmptyField);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(KernargAbiError::InvalidAlignment { alignment });
        }
        if !offset.is_multiple_of(alignment) {
            return Err(KernargAbiError::MisalignedField { offset, alignment });
        }
        offset
            .checked_add(size)
            .ok_or(KernargAbiError::FieldAddressOverflow { offset, size })?;
        Ok(Self {
            offset,
            size,
            alignment,
        })
    }

    pub const fn offset(self) -> u32 {
        self.offset
    }

    pub const fn size(self) -> u32 {
        self.size
    }

    pub const fn alignment(self) -> u32 {
        self.alignment
    }

    pub const fn end(self) -> u32 {
        self.offset + self.size
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernargAbiHash(Sha256Digest);

impl KernargAbiHash {
    pub const fn digest(self) -> Sha256Digest {
        self.0
    }
}

impl fmt::Debug for KernargAbiHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("KernargAbiHash")
            .field(&self.0)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernargAbi {
    segment_size: u32,
    segment_alignment: u32,
    fields: Vec<KernargField>,
    hash: KernargAbiHash,
}

impl KernargAbi {
    pub fn new(
        segment_size: u32,
        segment_alignment: u32,
        fields: impl IntoIterator<Item = KernargField>,
    ) -> Result<Self, KernargAbiError> {
        if segment_alignment == 0 || !segment_alignment.is_power_of_two() {
            return Err(KernargAbiError::InvalidSegmentAlignment {
                alignment: segment_alignment,
            });
        }
        let fields = fields.into_iter().collect::<Vec<_>>();
        for (index, field) in fields.iter().copied().enumerate() {
            if field.alignment > segment_alignment {
                return Err(KernargAbiError::FieldExceedsSegmentAlignment {
                    index,
                    field_alignment: field.alignment,
                    segment_alignment,
                });
            }
            if field.end() > segment_size {
                return Err(KernargAbiError::FieldOutOfBounds {
                    index,
                    end: field.end(),
                    segment_size,
                });
            }
            for (other_index, other) in fields.iter().copied().enumerate().take(index) {
                if field.offset < other.end() && other.offset < field.end() {
                    return Err(KernargAbiError::OverlappingFields {
                        first: other_index,
                        second: index,
                    });
                }
            }
        }
        let hash = hash_abi(segment_size, segment_alignment, &fields);
        Ok(Self {
            segment_size,
            segment_alignment,
            fields,
            hash,
        })
    }

    pub const fn segment_size(&self) -> u32 {
        self.segment_size
    }

    pub const fn segment_alignment(&self) -> u32 {
        self.segment_alignment
    }

    pub fn fields(&self) -> &[KernargField] {
        &self.fields
    }

    pub const fn hash(&self) -> KernargAbiHash {
        self.hash
    }

    pub fn validate_arguments(&self, arguments: &[KernelArg]) -> Result<(), KernargAbiError> {
        if arguments.len() != self.fields.len() {
            return Err(KernargAbiError::ArgumentCount {
                fields: self.fields.len(),
                arguments: arguments.len(),
            });
        }
        for (index, (argument, field)) in arguments.iter().zip(&self.fields).enumerate() {
            let actual = match argument {
                KernelArg::Scalar(bytes) => u32::try_from(bytes.len()).unwrap_or(u32::MAX),
                KernelArg::ScalarSlot { size, .. } => *size,
                KernelArg::Resource { .. } => DEVICE_POINTER_BYTES,
            };
            if actual != field.size {
                return Err(KernargAbiError::ArgumentSize {
                    index,
                    expected: field.size,
                    actual,
                });
            }
        }
        Ok(())
    }
}

fn hash_abi(segment_size: u32, segment_alignment: u32, fields: &[KernargField]) -> KernargAbiHash {
    let mut hasher = Sha256::new();
    hasher.update(b"redline-kernarg-abi-v1\0");
    hasher.update(segment_size.to_le_bytes());
    hasher.update(segment_alignment.to_le_bytes());
    hasher.update((fields.len() as u64).to_le_bytes());
    for field in fields {
        hasher.update(field.offset.to_le_bytes());
        hasher.update(field.size.to_le_bytes());
        hasher.update(field.alignment.to_le_bytes());
    }
    KernargAbiHash(Sha256Digest(hasher.finalize().into()))
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelArtifactIdentity {
    code_object: Sha256Digest,
    symbol_text: Sha256Digest,
    generation: u64,
}

impl KernelArtifactIdentity {
    pub const fn new(
        code_object: Sha256Digest,
        symbol_text: Sha256Digest,
        generation: u64,
    ) -> Self {
        Self {
            code_object,
            symbol_text,
            generation,
        }
    }

    pub fn from_bytes(code_object: &[u8], symbol_text: &[u8], generation: u64) -> Self {
        Self::new(
            Sha256Digest::digest(code_object),
            Sha256Digest::digest(symbol_text),
            generation,
        )
    }

    pub const fn code_object(self) -> Sha256Digest {
        self.code_object
    }

    pub const fn symbol_text(self) -> Sha256Digest {
        self.symbol_text
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArtifactCatalog {
    entries: BTreeMap<String, KernelArtifactIdentity>,
}

impl ArtifactCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        kernel: impl Into<String>,
        identity: KernelArtifactIdentity,
    ) -> Result<(), ArtifactCatalogError> {
        let kernel = kernel.into();
        if kernel.trim().is_empty() {
            return Err(ArtifactCatalogError::EmptyKernel);
        }
        self.entries.insert(kernel, identity);
        Ok(())
    }

    pub fn get(&self, kernel: &str) -> Option<KernelArtifactIdentity> {
        self.entries.get(kernel).copied()
    }
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ArtifactCatalogError {
    #[error("kernel key is empty")]
    EmptyKernel,
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum KernargAbiError {
    #[error("kernarg field size is zero")]
    EmptyField,
    #[error("kernarg field alignment {alignment} is not a nonzero power of two")]
    InvalidAlignment { alignment: u32 },
    #[error("kernarg segment alignment {alignment} is not a nonzero power of two")]
    InvalidSegmentAlignment { alignment: u32 },
    #[error("kernarg field at {offset} is not aligned to {alignment}")]
    MisalignedField { offset: u32, alignment: u32 },
    #[error("kernarg field range {offset}+{size} overflows")]
    FieldAddressOverflow { offset: u32, size: u32 },
    #[error(
        "kernarg field {index} alignment {field_alignment} exceeds segment alignment {segment_alignment}"
    )]
    FieldExceedsSegmentAlignment {
        index: usize,
        field_alignment: u32,
        segment_alignment: u32,
    },
    #[error("kernarg field {index} ends at {end}, beyond segment size {segment_size}")]
    FieldOutOfBounds {
        index: usize,
        end: u32,
        segment_size: u32,
    },
    #[error("kernarg fields {first} and {second} overlap")]
    OverlappingFields { first: usize, second: usize },
    #[error("kernarg ABI has {fields} fields but launch has {arguments} arguments")]
    ArgumentCount { fields: usize, arguments: usize },
    #[error("kernarg argument {index} is {actual} bytes, expected {expected}")]
    ArgumentSize {
        index: usize,
        expected: u32,
        actual: u32,
    },
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn overlapping_kernarg_fields_are_rejected(
            offset in 0_u32..1_024,
            first_size in 2_u32..128,
            overlap_seed in any::<u32>(),
            second_size in 1_u32..128,
        ) {
            let second_offset = offset + overlap_seed % first_size;
            let first = KernargField::new(offset, first_size, 1).unwrap();
            let second = KernargField::new(second_offset, second_size, 1).unwrap();
            let segment_size = first.end().max(second.end());

            assert_eq!(
                KernargAbi::new(segment_size, 1, [first, second]).unwrap_err(),
                KernargAbiError::OverlappingFields {
                    first: 0,
                    second: 1,
                },
            );
        }

        #[test]
        fn identical_kernarg_layouts_have_stable_hashes(
            field_size in 1_u32..256,
            trailing_padding in 0_u32..256,
        ) {
            let field = KernargField::new(0, field_size, 1).unwrap();
            let segment_size = field_size + trailing_padding;
            let first = KernargAbi::new(segment_size, 1, [field]).unwrap();
            let second = KernargAbi::new(segment_size, 1, [field]).unwrap();

            prop_assert_eq!(first.hash(), second.hash());
        }
    }
}
