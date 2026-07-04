//! W3a — minimal AXLF (`xclbin2`) container parser: enough to enumerate sections
//! and pull out the AIE partition / PDI / metadata that CREATE_HWCTX + CONFIG_HWCTX
//! need for the W4A8 kernel wire-in (see docs/npu/wire-in-amdxdna-command-submission.md).
//!
//! Layout pinned against `/usr/include/xrt/detail/xclbin.h`:
//! `axlf` = magic[8] + sig_len(4) + reserved(28) + keyBlock(256) + uniqueId(8) +
//! `axlf_header`(152) + `axlf_section_header`[num] (40 each). So `m_numSections`
//! sits at file offset 448 and the section table at 456. Each section header is
//! kind(u32)@0, name[16]@4, offset(u64)@24, size(u64)@32.
//!
//! Pure byte parsing — no ioctl, no allocation of the section data (returns
//! borrowed slices), works on every target.

/// AXLF section kinds we care about (`enum axlf_section_kind`).
pub const KIND_EMBEDDED_METADATA: u32 = 2;
pub const KIND_PDI: u32 = 18;
pub const KIND_PARTITION_METADATA: u32 = 20;
pub const KIND_AIE_METADATA: u32 = 25;
pub const KIND_AIE_RESOURCES: u32 = 29;
/// The AIE partition. For mlir-aie xclbins this section carries the partition
/// metadata (column count, ...) **and** the embedded PDI — there is no standalone
/// PDI section, so W3b reads the partition/PDI out of here.
pub const KIND_AIE_PARTITION: u32 = 32;

const MAGIC: &[u8; 8] = b"xclbin2\0";
const NUM_SECTIONS_OFF: usize = 448;
const SECTION_TABLE_OFF: usize = 456;
const SECTION_HDR_SIZE: usize = 40;

/// One AXLF section: its kind, name, and the byte range of its data in the file.
#[derive(Debug, Clone)]
pub struct Section {
    pub kind: u32,
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Errors parsing an AXLF container.
#[derive(Debug)]
pub enum XclbinError {
    /// Buffer too small or magic mismatch.
    NotAxlf,
    /// A section header or its data range runs past the buffer.
    Truncated,
}

impl std::fmt::Display for XclbinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XclbinError::NotAxlf => write!(f, "not an xclbin2/AXLF container"),
            XclbinError::Truncated => write!(f, "xclbin section table/data is truncated"),
        }
    }
}

impl std::error::Error for XclbinError {}

/// A parsed AXLF container borrowing the file bytes.
pub struct Axlf<'a> {
    bytes: &'a [u8],
    /// Section directory (kind/name/offset/size).
    pub sections: Vec<Section>,
}

fn rd_u16(b: &[u8], off: usize) -> Option<u16> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
}
fn rd_u32(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}
fn rd_u64(b: &[u8], off: usize) -> Option<u64> {
    b.get(off..off + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
}

impl<'a> Axlf<'a> {
    /// Parse the section directory of an AXLF buffer.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, XclbinError> {
        if bytes.len() < SECTION_TABLE_OFF || &bytes[0..8] != MAGIC {
            return Err(XclbinError::NotAxlf);
        }
        let num = rd_u32(bytes, NUM_SECTIONS_OFF).ok_or(XclbinError::NotAxlf)? as usize;
        // Guard against a corrupt/huge count (XCLBIN_MAX_NUM_SECTION = 0x10000).
        if num > 0x10000 {
            return Err(XclbinError::NotAxlf);
        }
        let mut sections = Vec::with_capacity(num);
        for i in 0..num {
            let h = SECTION_TABLE_OFF + i * SECTION_HDR_SIZE;
            let kind = rd_u32(bytes, h).ok_or(XclbinError::Truncated)?;
            let name_raw = bytes.get(h + 4..h + 20).ok_or(XclbinError::Truncated)?;
            let name = name_raw
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as char)
                .collect::<String>();
            let offset = rd_u64(bytes, h + 24).ok_or(XclbinError::Truncated)? as usize;
            let size = rd_u64(bytes, h + 32).ok_or(XclbinError::Truncated)? as usize;
            // A section's data range must lie within the buffer.
            if offset.checked_add(size).map(|e| e > bytes.len()) != Some(false) {
                return Err(XclbinError::Truncated);
            }
            sections.push(Section {
                kind,
                name,
                offset,
                size,
            });
        }
        Ok(Axlf { bytes, sections })
    }

    /// The data bytes of the first section of `kind`, if present.
    pub fn section(&self, kind: u32) -> Option<&'a [u8]> {
        let s = self.sections.iter().find(|s| s.kind == kind)?;
        self.bytes.get(s.offset..s.offset + s.size)
    }

    /// Parse the AIE_PARTITION section into the column count + start columns + the
    /// embedded PDI image (the fields CREATE_HWCTX / CONFIG_HWCTX need). Layout per
    /// `struct aie_partition` (184 B): `info.column_width` u16 @32, `info.start_columns`
    /// array_offset @40, `aie_pdi` array_offset @120. Each `aie_pdi` entry (96 B) has
    /// `pdi_image` array_offset @+16. All array offsets are relative to the section.
    pub fn aie_partition(&self) -> Option<AiePartition<'a>> {
        let sec = self.section(KIND_AIE_PARTITION)?;
        let column_width = rd_u16(sec, 32)?;

        let n_cols = rd_u32(sec, 40)? as usize;
        let cols_off = rd_u32(sec, 44)? as usize;
        let mut start_columns = Vec::with_capacity(n_cols);
        for i in 0..n_cols {
            start_columns.push(rd_u16(sec, cols_off + i * 2)?);
        }

        let n_pdi = rd_u32(sec, 120)? as usize;
        let pdi_arr_off = rd_u32(sec, 124)? as usize;
        if n_pdi == 0 {
            return None;
        }
        // First PDI entry; pdi_image = {size, offset} at entry+16.
        let pdi_size = rd_u32(sec, pdi_arr_off + 16)? as usize;
        let pdi_off = rd_u32(sec, pdi_arr_off + 20)? as usize;
        let pdi = sec.get(pdi_off..pdi_off + pdi_size)?;

        Some(AiePartition {
            column_width,
            start_columns,
            pdi,
        })
    }
}

/// The parsed AIE_PARTITION: partition width, its start columns, and the embedded
/// PDI image (the compiled tile program the driver loads).
pub struct AiePartition<'a> {
    /// Number of AIE columns this partition occupies.
    pub column_width: u16,
    /// Start-column identifiers for the partition placement.
    pub start_columns: Vec<u16>,
    /// The first PDI image bytes (mlir-aie emits one PDI per xclbin).
    pub pdi: &'a [u8],
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal AXLF with one PDI section pointing at 4 bytes of data.
    fn synth() -> Vec<u8> {
        let data_off = SECTION_TABLE_OFF + SECTION_HDR_SIZE; // one section header
        let mut b = vec![0u8; data_off + 4];
        b[0..8].copy_from_slice(MAGIC);
        b[NUM_SECTIONS_OFF..NUM_SECTIONS_OFF + 4].copy_from_slice(&1u32.to_le_bytes());
        let h = SECTION_TABLE_OFF;
        b[h..h + 4].copy_from_slice(&KIND_PDI.to_le_bytes());
        b[h + 4..h + 8].copy_from_slice(b"pdi\0");
        b[h + 24..h + 32].copy_from_slice(&(data_off as u64).to_le_bytes());
        b[h + 32..h + 40].copy_from_slice(&4u64.to_le_bytes());
        b[data_off..data_off + 4].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        b
    }

    #[test]
    fn parses_sections_and_finds_pdi() {
        let b = synth();
        let axlf = Axlf::parse(&b).expect("parse");
        assert_eq!(axlf.sections.len(), 1);
        assert_eq!(axlf.sections[0].kind, KIND_PDI);
        assert_eq!(axlf.sections[0].name, "pdi");
        assert_eq!(axlf.section(KIND_PDI), Some(&[0xde, 0xad, 0xbe, 0xef][..]));
        assert!(axlf.section(KIND_AIE_METADATA).is_none());
    }

    #[test]
    fn rejects_non_axlf() {
        assert!(matches!(
            Axlf::parse(&[0u8; 512]),
            Err(XclbinError::NotAxlf)
        ));
    }
}
