use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use needletail::{parse_fastx_file, Sequence};
use std::collections::HashMap;

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

#[pymodule]
fn kmer_counter_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_kmer_rs, m)?)?;
    Ok(())
}