import unittest
import os
import tempfile
from kmer_counter_rs import extract_kmer_rs


class TestExtractKmerRs(unittest.TestCase):
    def _create_temp_fasta_file(self, sequences):
        """
        Helper function to create a temporary FASTA file.
        sequences: A list of tuples, where each tuple is (header, sequence_string).
        Returns the path to the temporary FASTA file.
        """
        fd, temp_file_path = tempfile.mkstemp(suffix=".fasta")
        with os.fdopen(fd, "w") as tmpfile:
            for header, seq_str in sequences:
                tmpfile.write(f">{header}\n")
                tmpfile.write(f"{seq_str}\n")
        return temp_file_path

    def test_empty_file(self):
        temp_fasta = self._create_temp_fasta_file([])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {})
        finally:
            os.remove(temp_fasta)

    def test_short_sequence_less_than_k(self):
        # Sequence "ACG" is shorter than k=4
        temp_fasta = self._create_temp_fasta_file([("seq1", "ACG")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {})
        finally:
            os.remove(temp_fasta)

    def test_single_kmer(self):
        # "AGCT" -> canonical is "AGCT"
        temp_fasta = self._create_temp_fasta_file([("seq1", "AGCT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {b"AGCT": 1})
        finally:
            os.remove(temp_fasta)

    def test_multiple_same_kmer_overlapping(self):
        # "AAAAA" -> "AAAA" (1st), "AAAA" (2nd)
        temp_fasta = self._create_temp_fasta_file([("seq1", "AAAAA")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {b"AAAA": 2})
        finally:
            os.remove(temp_fasta)

    def test_sequence_exactly_k(self):
        # Sequence "TTTT" -> canonical is "AAAA"
        temp_fasta = self._create_temp_fasta_file([("seq1", "TTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {b"AAAA": 1})
        finally:
            os.remove(temp_fasta)

    def test_different_kmers(self):
        # "AGCTTTCA" -> AGCT, GCTT, CTTT, TTTC, TTCA
        # Canonical: AGCT, GCTT, CTTT, AAAG (revcomp of TTTC), TGAA (revcomp of TTCA)
        # Needletail's canonical_kmers:
        # AGCT -> AGCT
        # GCTT -> GCTT
        # CTTT -> CTTT
        # TTTC -> AAAG (canonical)
        # TTCA -> TGAA (canonical)
        temp_fasta = self._create_temp_fasta_file([("seq1", "AGCTTTCA")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            # AGCT -> AGCT
            # GCTT -> AAGC (canonical)
            # CTTT -> AAAG (canonical)
            # TTTC -> AAAG (canonical)
            # TTCA -> TGAA (canonical)
            expected = {
                b"AGCT": 1,
                b"AAGC": 1,
                b"AAAG": 2, # CTTT and TTTC both map to AAAG
                b"TGAA": 1,
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_canonical_kmers_simple_reverse_complement(self):
        # "TTTT" is reverse complement of "AAAA", canonical is "AAAA"
        temp_fasta = self._create_temp_fasta_file([("seq1", "TTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {b"AAAA": 1})
        finally:
            os.remove(temp_fasta)

    def test_canonical_kmers_mixed(self):
        # "ACGT" (canonical ACGT) and "TTTT" (canonical AAAA)
        # "ACGTTTTT" -> ACGT, CGTT, GTTT, TTTT, TTTT
        # Canonical: ACGT, AACG, AAAC, AAAA, AAAA
        temp_fasta = self._create_temp_fasta_file([("seq1", "ACGTTTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            expected = {
                b"ACGT": 1,
                b"AACG": 1,
                b"AAAC": 1,
                b"AAAA": 2,
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_multiple_sequences_in_file(self):
        temp_fasta = self._create_temp_fasta_file([
            ("seq1", "AAAAA"),  # AAAA: 2
            ("seq2", "GATTACA"),  # GATT, ATTA, TTAC, TACA -> GATT, TAAT, GTAA, TACA
            # GATT:1, TAAT:1, GTAA:1, TACA:1
        ])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            expected = {
                b"AAAA": 2,
                b"AATC": 1, # GATT (rev TGTA) -> AATC (rc of GATT, which is TGTA's rc)
                b"ATTA": 1, # ATTA (rev TAAT) -> ATTA
                b"GTAA": 1, # TTAC (rev GTAA) -> GTAA
                b"TACA": 1, # TACA (rev TGTA) -> TACA
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_sequence_with_n(self):
        # Needletail should skip kmers with N
        # "AAANTTT" -> AAA (skip AAN, ANT, NTT), TTT
        # Canonical: AAA, AAA
        temp_fasta = self._create_temp_fasta_file([
            ("seq1", "AAANTTTT")
        ])  # AAAN, AANT, ANTT, NTTT, TTTT
        # Kmers from needletail: TTTT (AAAA)
        # If k=4, AAAN, AANT, ANTT, NTTT are skipped by needletail.
        # Only TTTT processed.
        try:
            # This test confirms original behavior: N-containing kmers are skipped by needletail
            result = extract_kmer_rs(temp_fasta, 4, False)
            self.assertEqual(result, {b"AAAA": 1})
        finally:
            os.remove(temp_fasta)

    def test_sequence_with_n_process_n_true(self):
        # Sequence "AAANTTTT", k=4
        # Kmers: AAAN, AANT, ANTT, NTTT, TTTT
        # Canonical (N maps to N, A=65, C=67, G=71, N=78, T=84):
        # 1. AAAN (rc NTTT) -> AAAN (A < N)
        # 2. AANT (rc ANTT) -> AANT (A < N for 2nd char of rc ANTT)
        # 3. ANTT (rc AANT) -> AANT (A < N for 1st char of ANTT)
        # 4. NTTT (rc AAAN) -> AAAN (A < N)
        # 5. TTTT (rc AAAA) -> AAAA (A < T)
        temp_fasta = self._create_temp_fasta_file([("seq1", "AAANTTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, True)
            expected = {
                b"AAAN": 2,  # From AAAN and NTTT
                b"AANT": 2,  # From AANT and ANTT
                b"AAAA": 1,  # From TTTT
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_sequence_with_n_process_n_false(self):
        # Verify again that process_n_kmers=False skips N-containing kmers
        temp_fasta = self._create_temp_fasta_file([("seq1", "AAANTTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            expected = {
                b"AAAA": 1,  # Only TTTT (-> AAAA) should be found
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_multiple_sequences_with_n_process_n_true(self):
        # Seq1: "ACGT", k=3 -> ACG, CGT. Canonical: ACG, ACG. -> {ACG: 2}
        # Seq2: "ANNA", k=3 -> ANN, NNA.
        #   ANN (rc NNT). Canonical: ANN (A < N)
        #   NNA (rc TNN). Canonical: NNA (N < T, A < N for NNA vs ANN) -> ANN (A < N)
        # Result: {ACG: 2, ANN: 2}
        temp_fasta = self._create_temp_fasta_file([
            ("seq1", "ACGT"),
            ("seq2", "ANNA")
        ])
        try:
            result = extract_kmer_rs(temp_fasta, 3, True)
            # ACGT (k=3):
            # ACG (rc CGT) -> ACG
            # CGT (rc ACG) -> ACG
            # ANNA (k=3):
            # ANN (rc NNT) -> ANN (A < N)
            # NNA (rc TNN). Canonical of NNA and ANN. ANN is smaller (A vs N).
            #   NNA vs ANN. N vs A. ANN is smaller.
            #   So, for "NNA", its canonical is "ANN".
            expected = {
                b"ACG": 2,
                b"ANN": 2,
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_long_sequence_repeats(self):
        temp_fasta = self._create_temp_fasta_file([("seq1", "AGCTAGCTAGCT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4, False)
            expected = {
                b"AGCT": 3,
                b"GCTA": 4,
                b"CTAG": 2,
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)


if __name__ == "__main__":
    unittest.main()
