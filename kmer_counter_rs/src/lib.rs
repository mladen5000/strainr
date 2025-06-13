use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyBytes;
// Use the same imports as extract_kmer_rs for Sequence
use needletail::{parse_fastx_file, Sequence};
use std::collections::HashMap;
use log::{info, debug};
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logging() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

#[pyfunction]
fn extract_kmer_rs(file_path: &str) -> PyResult<HashMap<Vec<u8>, usize>> {
    let mut reader = match parse_fastx_file(file_path) {
        Ok(r) => r,
        Err(e) => {
            // Attempt to differentiate errors. If it's about reading initial bytes for format detection,
            // it might be an empty or malformed file.
            if e.to_string().contains("Failed to read the first two bytes") || e.to_string().contains("empty file") {
                // For genuinely empty or unreadable-as-fasta files, return empty map.
                return Ok(HashMap::new());
            }
            // For other IO errors, propagate them.
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open or parse file: {}", e)));
        }
    };
    let mut kmer_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut processed_any_record = false;

    while let Some(record) = reader.next() {
        processed_any_record = true;
        let seqrec = record.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid record: {}", e)))?;
        let norm_seq = seqrec.normalize(false);
        let rc = norm_seq.reverse_complement();
        for (_, kmer, _) in norm_seq.canonical_kmers(4, &rc) {
            let kmer_vec = kmer.to_vec();
            *kmer_counts.entry(kmer_vec).or_insert(0) += 1;
        }
    }

    if !processed_any_record {
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
            _ => b'N', // For any other character, treat as N
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
fn extract_kmers_rs(sequence_bytes: &Bound<'_, PyBytes>, k: usize) -> PyResult<Vec<Vec<u8>>> {
    init_logging();
    
    let seq_bytes = sequence_bytes.as_bytes();
    let mut kmers = Vec::new();
    
    if k == 0 { // k=0 is problematic, return empty or error? Let's return empty.
        debug!("k cannot be 0, returning empty list.");
        return Ok(kmers);
    }
    if seq_bytes.len() < k {
        debug!("Sequence too short ({} bp) for k={}", seq_bytes.len(), k);
        return Ok(kmers);
    }
    
    debug!("Manually processing sequence of {} bytes with k={}", seq_bytes.len(), k);
    
    let mut total_kmers_considered = 0;
    let mut actual_kmers_extracted = 0;
    let mut num_skipped_kmers = 0;

    for i in 0..=(seq_bytes.len() - k) {
        let kmer_slice = &seq_bytes[i..i+k];
        total_kmers_considered += 1;

        if !is_valid_dna(kmer_slice) {
            num_skipped_kmers += 1;
            // Optionally log each skipped k-mer, but this can be very verbose:
            // debug!("Skipping k-mer with non-DNA char: {:?}", std::str::from_utf8(kmer_slice).unwrap_or("invalid utf8"));
            continue;
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
            debug!("Processed {} valid k-mers so far ({} considered)", actual_kmers_extracted, total_kmers_considered);
        }
    }
    
    if num_skipped_kmers > 0 {
        debug!("Skipped {} k-mers containing non-ACGTacgt characters.", num_skipped_kmers);
    }

    let expected_kmers_if_all_valid = if seq_bytes.len() >= k { seq_bytes.len() - k + 1 } else { 0 };
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