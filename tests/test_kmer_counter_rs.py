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
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {})
        finally:
            os.remove(temp_fasta)

    def test_short_sequence_less_than_k(self):
        # Sequence "ACG" is shorter than k=4
        temp_fasta = self._create_temp_fasta_file([("seq1", "ACG")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {})
        finally:
            os.remove(temp_fasta)

    def test_single_kmer(self):
        # "AGCT" -> canonical is "AGCT"
        temp_fasta = self._create_temp_fasta_file([("seq1", "AGCT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {b"AGCT": 1})
        finally:
            os.remove(temp_fasta)

    def test_multiple_same_kmer_overlapping(self):
        # "AAAAA" -> "AAAA" (1st), "AAAA" (2nd)
        temp_fasta = self._create_temp_fasta_file([("seq1", "AAAAA")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {b"AAAA": 2})
        finally:
            os.remove(temp_fasta)

    def test_sequence_exactly_k(self):
        # Sequence "TTTT" -> canonical is "AAAA"
        temp_fasta = self._create_temp_fasta_file([("seq1", "TTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
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
            result = extract_kmer_rs(temp_fasta, 4)
            expected = {
                b"AGCT": 1,
                b"AAGC": 1,  # Canonical of GCTT
                b"GAAA": 1,  # Canonical of CTTT
                b"AAAG": 1,  # Canonical of TTTC
                b"TGAA": 1,  # Canonical of TTCA
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)

    def test_canonical_kmers_simple_reverse_complement(self):
        # "TTTT" is reverse complement of "AAAA", canonical is "AAAA"
        temp_fasta = self._create_temp_fasta_file([("seq1", "TTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {b"AAAA": 1})
        finally:
            os.remove(temp_fasta)

    def test_canonical_kmers_mixed(self):
        # "ACGT" (canonical ACGT) and "TTTT" (canonical AAAA)
        # "ACGTTTTT" -> ACGT, CGTT, GTTT, TTTT, TTTT
        # Canonical: ACGT, AACG, AAAC, AAAA, AAAA
        temp_fasta = self._create_temp_fasta_file([("seq1", "ACGTTTTT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            expected = {
                b"ACGT": 1,  # ACGT
                b"AACG": 1,  # CGTT -> AACG
                b"AAAC": 1,  # GTTT -> AAAC
                b"AAAA": 2,  # TTTT, TTTT -> AAAA, AAAA
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
            result = extract_kmer_rs(temp_fasta, 4)
            expected = {
                b"AAAA": 2,
                b"AATC": 1,  # Canonical of GATT
                b"ATTA": 1,  # Canonical of ATTA
                b"GTAA": 1,  # Canonical of TTAC
                b"TACA": 1,  # Canonical of TACA
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
            result = extract_kmer_rs(temp_fasta, 4)
            self.assertEqual(result, {b"AAAA": 1})  # Only TTTT -> AAAA
        finally:
            os.remove(temp_fasta)

    def test_long_sequence_repeats(self):
        # "AGCTAGCTAGCT" -> AGCT, GCTA, CTAG, TAGC, AGCT, GCTA, CTAG, TAGC, AGCT
        # Canonical: AGCT, AGCT, AGCT, CTAG, CTAG, CTAG, GCTA, GCTA, TAGC
        # AGCT: 3, CTAG: 2, GCTA: 2, TAGC: 2 (This needs careful check for canonical)
        # AGCT (AGCT)
        # GCTA (GCTA)
        # CTAG (CTAG)
        # TAGC (TAGC)
        # AGCT (AGCT)
        # GCTA (GCTA)
        # CTAG (CTAG)
        # TAGC (TAGC)
        # AGCT (AGCT)
        temp_fasta = self._create_temp_fasta_file([("seq1", "AGCTAGCTAGCT")])
        try:
            result = extract_kmer_rs(temp_fasta, 4)
            expected = {
                b"AGCT": 3,
                b"GCTA": 4,  # TAGC's canonical is GCTA
                b"CTAG": 2,
            }
            self.assertEqual(result, expected)
        finally:
            os.remove(temp_fasta)


if __name__ == "__main__":
    unittest.main()
