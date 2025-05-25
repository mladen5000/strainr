use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use needletail::{parse_fastx_file, Sequence};

#[pyfunction]
fn extract_kmer_rs(file_path: &str) -> PyResult<usize> {
    let mut reader = parse_fastx_file(file_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
    let mut count = 0;

    while let Some(record) = reader.next() {
        let seqrec = record.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid record: {}", e)))?;
        let norm_seq = seqrec.normalize(false);
        let rc = norm_seq.reverse_complement();
        for (_, kmer, _) in norm_seq.canonical_kmers(4, &rc) {
            if kmer == b"AAAA" { //TODO: Make this count the kmers and output them
                count += 1;
            }
        }
    }

    Ok(count)
}

#[pymodule]
fn kmer_counter_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_kmer_rs, m)?)?;
    Ok(())
}