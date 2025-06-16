import pytest
import pathlib
from strainr.build_db import BuildDBArgs, DatabaseBuilder

@pytest.fixture
def db_builder(tmp_path: pathlib.Path):
    """
    Pytest fixture to set up a DatabaseBuilder instance with mock arguments
    and a temporary directory structure for testing.
    """
    custom_genomes_dir = tmp_path / "custom_genomes"
    custom_genomes_dir.mkdir()

    # Create a dummy genome file inside custom_genomes_dir for BuildDBArgs validation if needed
    # (BuildDBArgs itself doesn't check for files inside custom_dir, but good practice for some tests)
    (custom_genomes_dir / "dummy_genome.fna").touch()

    args = BuildDBArgs(
        custom=custom_genomes_dir,
        kmerlen=5,  # Using a small, testable k-mer length
        out="test_db_output", # Prefix for output files/dirs
        procs=1, # Keep it simple for testing
        # Add other essential defaults if DatabaseBuilder requires them
        assembly_levels="complete", # Default, but good to be explicit
        source="refseq", # Default, but good to be explicit
        # taxid, assembly_accessions, genus are None due to 'custom' being used
    )

    builder_instance = DatabaseBuilder(args=args)

    # The DatabaseBuilder creates an output directory like <out>_files
    # Ensure this directory is also within tmp_path for isolation
    builder_instance.output_db_dir = tmp_path / (args.out + "_files")
    builder_instance.output_db_dir.mkdir(parents=True, exist_ok=True)

    # Also, DatabaseBuilder.base_path defaults to cwd. For tests, make it tmp_path.
    builder_instance.base_path = tmp_path

    yield builder_instance, tmp_path, args
    # No explicit teardown needed as tmp_path handles cleanup
