use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
// Use the same imports as extract_kmer_rs for Sequence
use log::{debug, error, info};
use needletail::{parse_fastx_file, Sequence};
use std::collections::HashMap;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logging() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

#[pyfunction]
fn extract_kmer_rs(file_path: &str, k: u8) -> PyResult<HashMap<Vec<u8>, usize>> {
    init_logging();
    info!("Starting extract_kmer_rs for file: {}", file_path);
    let mut reader = match parse_fastx_file(file_path) {
        Ok(r) => {
            info!("Successfully opened file: {}", file_path);
            r
        }
        Err(e) => {
            info!("Failed to open file: {}. Error: {}", file_path, e);
            // Attempt to differentiate errors. If it's about reading initial bytes for format detection,
            // it might be an empty or malformed file.
            if e.to_string().contains("Failed to read the first two bytes")
                || e.to_string().contains("empty file")
            {
                info!(
                    "File appears empty or unreadable as FASTA: {}. Returning empty map.",
                    file_path
                );
                // For genuinely empty or unreadable-as-fasta files, return empty map.
                return Ok(HashMap::new());
            }
            // For other IO errors, propagate them.
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open or parse file: {}",
                e
            )));
        }
    };
    let mut kmer_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut processed_any_record = false;
    let mut total_records = 0;
    let mut total_kmers = 0;

    while let Some(record) = reader.next() {
        processed_any_record = true;
        total_records += 1;
        let seqrec = record.map_err(|e| {
            error!("Invalid record in file {}: {}", file_path, e);
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid record: {}", e))
        })?;
        let norm_seq = seqrec.normalize(false);
        let rc = norm_seq.reverse_complement();
        let mut record_kmers = 0;
        for (_, kmer, _) in norm_seq.canonical_kmers(k, &rc) {
            let kmer_vec = kmer.to_vec();
            *kmer_counts.entry(kmer_vec).or_insert(0) += 1;
            record_kmers += 1;
        }
        total_kmers += record_kmers;
        debug!(
            "Processed record {}: {} k-mers",
            total_records, record_kmers
        );
    }

    info!(
        "extract_kmer_rs finished for file: {}. Records: {}, Total k-mers: {}",
        file_path, total_records, total_kmers
    );

    if !processed_any_record {
        info!(
            "No records found in file: {}. Returning empty map.",
            file_path
        );
        // If the file was valid FASTA but contained no sequences, or if parse_fastx_file succeeded
        // but then no records were found (e.g. file with only comments after header).
        return Ok(HashMap::new());
    }

    Ok(kmer_counts)
}

// Helper function for reverse complement
fn reverse_complement_dna(dna: &[u8]) -> Vec<u8> {
    let mut rc = Vec::with_capacity(dna.len());
    for &base in dna.iter().rev() {
        rc.push(match base {
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' => b'A',
            b'N' | b'n' => b'N', // Keep Ns as Ns in reverse complement
            _ => b'N',           // For any other character, treat as N
        });
    }
    rc
}

// Helper function to check for valid DNA characters (ACGTacgt)
fn is_valid_dna(kmer_slice: &[u8]) -> bool {
    for &base in kmer_slice {
        match base {
            b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't' => (), // valid
            _ => return false, // invalid character found (e.g., 'N')
        }
    }
    true
}

#[pyfunction]
fn extract_kmers_rs(
    sequence_bytes: &Bound<'_, PyBytes>,
    k: usize,
    perform_strict_dna_check: bool,
) -> PyResult<Vec<Vec<u8>>> {
    init_logging();

    let seq_bytes = sequence_bytes.as_bytes();
    let mut kmers = Vec::new();

    if k == 0 {
        // k=0 is problematic, return empty or error? Let's return empty.
        debug!("k cannot be 0, returning empty list.");
        return Ok(kmers);
    }
    if seq_bytes.len() < k {
        debug!("Sequence too short ({} bp) for k={}", seq_bytes.len(), k);
        return Ok(kmers);
    }

    debug!(
        "Manually processing sequence of {} bytes with k={}",
        seq_bytes.len(),
        k
    );

    let mut total_kmers_considered = 0;
    let mut actual_kmers_extracted = 0;
    let mut num_skipped_kmers = 0;

    for i in 0..=(seq_bytes.len() - k) {
        let kmer_slice = &seq_bytes[i..i + k];
        total_kmers_considered += 1;

        if perform_strict_dna_check {
            if !is_valid_dna(kmer_slice) {
                num_skipped_kmers += 1;
                // Optionally log each skipped k-mer, but this can be very verbose:
                // debug!("Skipping k-mer with non-DNA char: {:?}", std::str::from_utf8(kmer_slice).unwrap_or("invalid utf8"));
                continue;
            }
        }

        let rc_kmer_vec = reverse_complement_dna(kmer_slice);

        // Determine canonical k-mer
        // Need to compare &[u8] with Vec<u8>. Can convert kmer_slice to Vec or rc_kmer_vec to slice.
        // Or, more efficiently, compare slices directly if possible.
        // kmer_slice vs rc_kmer_vec.as_slice()
        if kmer_slice <= rc_kmer_vec.as_slice() {
            kmers.push(kmer_slice.to_vec());
        } else {
            kmers.push(rc_kmer_vec);
        }
        actual_kmers_extracted += 1;

        if actual_kmers_extracted % 100000 == 0 && actual_kmers_extracted > 0 {
            debug!(
                "Processed {} valid k-mers so far ({} considered)",
                actual_kmers_extracted, total_kmers_considered
            );
        }
    }

    if num_skipped_kmers > 0 {
        debug!(
            "Skipped {} k-mers containing non-ACGTacgt characters.",
            num_skipped_kmers
        );
    }

    let expected_kmers_if_all_valid = if seq_bytes.len() >= k {
        seq_bytes.len() - k + 1
    } else {
        0
    };
    info!(
        "Extracted {} k-mers from {} bp sequence ({} windows considered, {} skipped, ~{} expected if all valid DNA)",
        kmers.len(),
        seq_bytes.len(),
        total_kmers_considered,
        num_skipped_kmers,
        expected_kmers_if_all_valid
    );

    Ok(kmers)
}

#[pyfunction]
fn enable_logging(level: Option<&str>) -> PyResult<()> {
    let log_level = level.unwrap_or("info");
    std::env::set_var("RUST_LOG", format!("kmer_counter_rs={}", log_level));
    init_logging();
    info!("Rust logging enabled at {} level", log_level);
    Ok(())
}

#[pymodule]
fn kmer_counter_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_kmer_rs, m)?)?;
    m.add_function(wrap_pyfunction!(extract_kmers_rs, m)?)?;
    m.add_function(wrap_pyfunction!(enable_logging, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyBytes;
    use pyo3::Python;

    #[test]
    fn test_extract_kmers_rs_with_n_no_strict_check() {
        Python::with_gil(|py| {
            let seq_str = "ACNGT"; // Sequence with 'N'
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 3;
            let perform_strict_dna_check = false;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed with error: {:?}",
                result.err()
            );
            let kmers = result.unwrap();

            // Expected k-mers (canonical):
            // "ACN" (original) vs "NGT" (rev-comp of "ACN") -> "ACN" (N is N, A < N, T > N - this depends on how 'N' is handled in RC and comparison)
            //   Let's assume 'N' is treated as itself in RC.
            //   ACN -> NGT. ACN vs NGT. 'A' vs 'N'. If 'A' < 'N', then ACN is canonical.
            // "CNG" (original) vs "NCA" (rev-comp of "CNG") -> "CNG"
            //   CNG -> NCA. CNG vs NCA. 'C' vs 'N'. If 'C' < 'N', then CNG is canonical.
            // "NGT" (original) vs "ACN" (rev-comp of "NGT") -> "ACN"
            //   NGT -> ACN. NGT vs ACN. 'N' vs 'A'. If 'A' < 'N', then ACN is canonical.
            // The reverse_complement_dna function treats N as N.
            // Comparison is lexicographical. b'A' (65), b'C' (67), b'G' (71), b'N' (78), b'T' (84)
            // So ACN vs NGT => ACN is canonical
            //    CNG vs NCA => CNG is canonical
            //    NGT vs ACN => ACN is canonical
            let expected_kmers: Vec<Vec<u8>> =
                vec![b"ACN".to_vec(), b"CNG".to_vec(), b"ACN".to_vec()];

            assert_eq!(
                kmers, expected_kmers,
                "K-mers with 'N' not handled as expected when strict check is off."
            );
        });
    }

    #[test]
    fn test_extract_kmers_rs_with_n_strict_check() {
        Python::with_gil(|py| {
            let seq_str = "ACNGT"; // Sequence with 'N'
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 3;
            let perform_strict_dna_check = true; // Strict check is ON

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed with error: {:?}",
                result.err()
            );
            let kmers = result.unwrap();

            // Expected: No k-mers should be extracted because 'N' makes them invalid with strict checking
            let expected_kmers: Vec<Vec<u8>> = Vec::new();

            assert_eq!(
                kmers, expected_kmers,
                "K-mers with 'N' should be skipped when strict check is on."
            );
        });
    }

    #[test]
    fn test_extract_kmers_rs_valid_dna_no_strict_check() {
        Python::with_gil(|py| {
            let seq_str = "ACGTA";
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 3;
            let perform_strict_dna_check = false;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed: {:?}",
                result.err()
            );
            let kmers = result.unwrap();

            // ACG (rev-comp CGT) -> ACG
            // CGT (rev-comp ACG) -> ACG
            // GTA (rev-comp TAC) -> GTA
            let expected_kmers: Vec<Vec<u8>> =
                vec![b"ACG".to_vec(), b"ACG".to_vec(), b"GTA".to_vec()];
            assert_eq!(kmers, expected_kmers);
        });
    }

    #[test]
    fn test_extract_kmers_rs_valid_dna_strict_check() {
        Python::with_gil(|py| {
            let seq_str = "ACGTA";
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 3;
            let perform_strict_dna_check = true;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed: {:?}",
                result.err()
            );
            let kmers = result.unwrap();
            // Expected k-mers are the same as without strict check because DNA is valid
            let expected_kmers: Vec<Vec<u8>> =
                vec![b"ACG".to_vec(), b"ACG".to_vec(), b"GTA".to_vec()];
            assert_eq!(kmers, expected_kmers);
        });
    }

    #[test]
    fn test_extract_kmers_rs_empty_sequence() {
        Python::with_gil(|py| {
            let seq_str = "";
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 3;
            let perform_strict_dna_check = false;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed: {:?}",
                result.err()
            );
            let kmers = result.unwrap();
            assert!(
                kmers.is_empty(),
                "Expected empty k-mer list for empty sequence"
            );
        });
    }

    #[test]
    fn test_extract_kmers_rs_k_too_large() {
        Python::with_gil(|py| {
            let seq_str = "ACG";
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 4;
            let perform_strict_dna_check = false;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed: {:?}",
                result.err()
            );
            let kmers = result.unwrap();
            assert!(
                kmers.is_empty(),
                "Expected empty k-mer list when k is larger than sequence length"
            );
        });
    }

    #[test]
    fn test_extract_kmers_rs_k_zero() {
        Python::with_gil(|py| {
            let seq_str = "ACGT";
            let py_bytes = PyBytes::new_bound(py, seq_str.as_bytes());
            let k = 0;
            let perform_strict_dna_check = false;

            let result = super::extract_kmers_rs(&py_bytes, k, perform_strict_dna_check);
            assert!(
                result.is_ok(),
                "extract_kmers_rs failed: {:?}",
                result.err()
            );
            let kmers = result.unwrap();
            assert!(kmers.is_empty(), "Expected empty k-mer list for k=0");
        });
    }

    #[test]
    fn test_is_valid_dna_helper() {
        assert!(super::is_valid_dna(b"ACGT"));
        assert!(super::is_valid_dna(b"acgt"));
        assert!(!super::is_valid_dna(b"ACGTN"));
        assert!(!super::is_valid_dna(b"ACGTX"));
        assert!(super::is_valid_dna(b"")); // Empty slice is valid
    }

    #[test]
    fn test_reverse_complement_dna_helper() {
        assert_eq!(super::reverse_complement_dna(b"ACGTN"), b"NACGT");
        assert_eq!(super::reverse_complement_dna(b"acgtn"), b"nacgt");
        assert_eq!(super::reverse_complement_dna(b""), b"");
        assert_eq!(super::reverse_complement_dna(b"GATTACA"), b"TGTAATC");
        assert_eq!(super::reverse_complement_dna(b"XxNn"), b"nNXX"); // X treated as N
    }
}
