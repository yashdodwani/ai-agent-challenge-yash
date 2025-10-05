import pytest
import pandas as pd
from pathlib import Path
import sys

# Add custom_parsers to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_parsers.icici_parser import parse


def test_icici_parser():
    """Test that parser output matches expected CSV."""
    # Parse PDF
    pdf_path = "data/icici/icici_sample.pdf"
    result_df = parse(pdf_path)

    # Load expected CSV
    expected_df = pd.read_csv("data/icici/icici_sample.csv")

    # Normalize data types
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        if col in expected_df.columns:
            expected_df[col] = pd.to_numeric(expected_df[col], errors='coerce')

    # Check shape
    assert result_df.shape == expected_df.shape, \
        f"Shape mismatch: got {result_df.shape}, expected {expected_df.shape}"

    # Check columns
    assert list(result_df.columns) == list(expected_df.columns), \
        f"Column mismatch: got {list(result_df.columns)}"

    # Check data equality (allowing for minor floating point differences)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), 
        expected_df.reset_index(drop=True),
        check_dtype=False,
        atol=0.01
    )

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_icici_parser()
