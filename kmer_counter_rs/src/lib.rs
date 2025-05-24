// kmer_counter_rs/src/lib.rs
use needletail::sequence::{Sequence, self as needletail_sequence_ops};
use pyo3::prelude::*;

#[pyfunction]
fn extract_kmers_rs(_py: Python, sequence: &[u8], k: usize) -> PyResult<Vec<Vec<u8>>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    if sequence.is_empty() {
        return Ok(Vec::new());
    }
    if sequence.len() < k {
        return Ok(Vec::new());
    }

    let normalized_seq_cow = needletail_sequence_ops::normalize(sequence, false);
    let normalized_seq = normalized_seq_cow.as_ref();

    let rc_normalized_seq = needletail_sequence_ops::reverse_complement(normalized_seq);

    let mut kmers_vec = Vec::new();
    let k_u8 = k as u8;
    if k_u8 == 0 && k > 0 {
         // k is too large for u8, or some other conversion issue
         // Consider returning a PyErr here for a clearer error message
         return Err(pyo3::exceptions::PyValueError::new_err(format!("kmer length {} is too large, max 255", k)));
    }

    if normalized_seq.len() >= k {
        for (_, kmer_slice, _) in needletail_sequence_ops::canonical_kmers(normalized_seq, &rc_normalized_seq, k_u8) {
            kmers_vec.push(kmer_slice.to_vec());
        }
    }
    Ok(kmers_vec)
}

#[pymodule]
fn kmer_counter_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_kmers_rs, m)?)?;
    Ok(())
}
