/// Gather kept rows from a row-major tensor's raw bytes. `shape[0]` is the
/// row count (e.g. experts); every row must be `bytes.len()/shape[0]` bytes.
/// Returns `(new_shape, gathered_bytes)`. Exact for row-independent quant.
pub fn gather_rows(
    shape: &[usize],
    bytes: &[u8],
    keep: &[u32],
) -> Result<(Vec<usize>, Vec<u8>), String> {
    let orig_rows = *shape.first().unwrap_or(&0);
    if orig_rows == 0 || bytes.len() % orig_rows != 0 {
        return Err(format!(
            "reap: row-gather: {orig_rows} rows don't divide {} bytes",
            bytes.len()
        ));
    }
    let rowstride = bytes.len() / orig_rows;
    let mut out = Vec::with_capacity(rowstride * keep.len());
    for &oe in keep {
        let oe = oe as usize;
        if oe >= orig_rows {
            return Err(format!(
                "reap: row-gather keep idx {oe} >= rows {orig_rows}"
            ));
        }
        out.extend_from_slice(&bytes[oe * rowstride..(oe + 1) * rowstride]);
    }
    let mut new_shape = shape.to_vec();
    new_shape[0] = keep.len();
    Ok((new_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathers_subset_in_order() {
        // 4 rows × 3 bytes
        let bytes: Vec<u8> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let (shape, out) = gather_rows(&[4, 3], &bytes, &[2, 0, 3]).unwrap();
        assert_eq!(shape, vec![3, 3]);
        assert_eq!(out, vec![2, 2, 2, 0, 0, 0, 3, 3, 3]);
    }

    #[test]
    fn identity_keep_is_byte_identical() {
        let bytes: Vec<u8> = (0..24).collect();
        let (_, out) = gather_rows(&[4, 6], &bytes, &[0, 1, 2, 3]).unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn errors_on_indivisible_rows() {
        let err = gather_rows(&[3, 2], &[0, 1, 2, 3, 4], &[0]).unwrap_err();
        assert!(err.contains("don't divide"), "got: {err}");
    }

    #[test]
    fn errors_on_out_of_range_keep() {
        let err = gather_rows(&[2, 2], &[0, 1, 2, 3], &[5]).unwrap_err();
        assert!(err.contains("keep idx 5 >= rows 2"), "got: {err}");
    }
}
