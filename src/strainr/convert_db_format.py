"""
Converts a StrainKmerDatabase from pickle format to Parquet format.

This script provides a command-line interface to load a pandas DataFrame
from a pickle file and save it to a Parquet file, ensuring the index
(which typically contains k-mers) is preserved.
"""

import argparse
import pandas as pd
from pathlib import Path
import pickle # Added import


def convert_pickle_to_parquet(pickle_path: Path, parquet_path: Path) -> None:
    """
    Loads a DataFrame from a pickle file and saves it to a Parquet file.

    Args:
        pickle_path: Path to the input pickle file.
        parquet_path: Path for the output Parquet file.

    Raises:
        FileNotFoundError: If the pickle_path does not exist.
        Exception: For other errors during file loading or saving.
    """
    try:
        print(f"Loading DataFrame from pickle file: {pickle_path}")
        df = pd.read_pickle(pickle_path)
        
        if not isinstance(df, pd.DataFrame):
            print(f"Error: The file {pickle_path} did not contain a pandas DataFrame.")
            return

        print(f"Saving DataFrame to Parquet file: {parquet_path}")
        # Ensure the directory for the parquet file exists
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=True)
        
        print(f"Successfully converted {pickle_path} to {parquet_path}")

    except FileNotFoundError:
        print(f"Error: Input pickle file not found at {pickle_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The pickle file {pickle_path} is empty or does not contain a valid DataFrame.")
    except pickle.UnpicklingError: 
        print(f"Error: Failed to unpickle data from {pickle_path}. The file may be corrupted or not a pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a StrainKmerDatabase from pickle format to Parquet format."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input pickle database file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path for the output Parquet database file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # It's good practice to avoid overwriting unintentionally, 
    # but for this script, we'll assume overwrite is fine or user manages it.
    # if output_path.exists():
    #     print(f"Warning: Output file {output_path} already exists. It will be overwritten.")

    convert_pickle_to_parquet(input_path, output_path)
