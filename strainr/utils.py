import functools
import gzip
from mimetypes import guess_type


def _open(infile):
    """
    Handle unknown file for gzip and non-gzip alike.

    Args:
        infile (str): The path to the input file.

    Returns:
        file_object: The file object for the opened file.
    """
    encoding = guess_type(str(infile))[1]  # uses file extension
    _open = functools.partial(gzip.open, mode="rt") if encoding == "gzip" else open
    file_object = _open(infile)
    return file_object


def generate_table(intermediate_results, strains):
    """
    Use the k-mer hits from classify in order to build a binning frame.

    Args:
        intermediate_results (dict): A dictionary containing the intermediate results of the classification.
        strains (list): A list of strains used as column names in the resulting DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame representing the binning frame with k-mer hits for each strain.
    """
    df = pd.DataFrame.from_dict(dict(intermediate_results)).T
    df.columns = strains
    return df
