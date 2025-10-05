import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
import dotenv


class ParserAgent:
    """Agent that autonomously generates and debugs bank statement parsers."""

    def __init__(self, api_key: str, max_attempts: int = 3):
        """
        Initialize the agent.

        Args:
            api_key: Google Gemini API key
            max_attempts: Maximum self-correction attempts
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_attempts = max_attempts
        self.conversation_history: List[Dict] = []

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history for context."""
        self.conversation_history.append({"role": role, "content": content})

    def _read_file(self, filepath: str) -> str:
        """Read file content safely."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading {filepath}: {str(e)}"

    def _write_file(self, filepath: str, content: str) -> bool:
        """Write content to file safely."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"❌ Error writing {filepath}: {e}")
            return False

    def _run_tests(self, test_file: str) -> Tuple[bool, str]:
        """
        Run pytest on the generated parser.

        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            result = subprocess.run(
                ['pytest', test_file, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + "\n" + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out"
        except Exception as e:
            return False, f"Test execution failed: {str(e)}"

    def _extract_code(self, response: str) -> str:
        """Extract Python code from markdown code blocks."""
        lines = response.split('\n')
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip().startswith('```') and in_code_block:
                in_code_block = False
                continue
            elif in_code_block:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # If no code blocks, return as-is (might be plain code)
        return response

    def plan(self, target: str, pdf_path: str, csv_path: str) -> str:
        """
        Phase 1: Plan the parser implementation.

        Args:
            target: Bank name (e.g., 'icici')
            pdf_path: Path to sample PDF
            csv_path: Path to expected CSV output

        Returns:
            Planning summary
        """
        print("📋 Phase 1: Planning...")

        # Read sample data
        csv_content = self._read_file(csv_path)

        prompt = f"""You are an expert Python developer. Analyze this bank statement parsing task:

TARGET BANK: {target}
PDF PATH: {pdf_path}
EXPECTED CSV OUTPUT (first 10 lines):
{chr(10).join(csv_content.split(chr(10))[:10])}

TASK: Create a parser function that:
1. Extracts tabular data from the PDF using tabula-py
2. Returns a pandas DataFrame with columns: Date, Description, Debit Amt, Credit Amt, Balance
3. Handles the specific format of {target.upper()} bank statements

Provide a brief plan (3-4 bullet points) on how to implement this parser.
Focus on: PDF extraction strategy, data cleaning steps, column mapping."""

        response = self.model.generate_content(prompt)
        plan = response.text
        self._add_to_history("assistant", f"Plan: {plan}")

        print(f"✅ Plan created:\n{plan}\n")
        return plan

    def generate_code(self, target: str, pdf_path: str, csv_path: str,
                      previous_error: Optional[str] = None) -> str:
        """
        Phase 2: Generate parser code.

        Args:
            target: Bank name
            pdf_path: Path to sample PDF
            csv_path: Path to expected CSV
            previous_error: Error from previous attempt (for self-correction)

        Returns:
            Generated Python code
        """
        print("🔨 Phase 2: Generating code...")

        csv_content = self._read_file(csv_path)

        base_prompt = f"""Generate a complete Python parser for {target.upper()} bank statements.

REQUIREMENTS:
1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
2. Use tabula-py to extract tables: tabula.read_pdf(pdf_path, pages='all', multiple_tables=False)
3. Return DataFrame with exact columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
4. Handle data type conversions (dates, floats)
5. Clean and normalize the data

EXPECTED OUTPUT FORMAT (CSV):
{chr(10).join(csv_content.split(chr(10))[:5])}

IMPORTANT:
- Import all required libraries (pandas, tabula)
- Add proper error handling
- Include docstring
- Keep it simple and robust
- Ensure numeric columns are float type
- Handle empty/NaN values properly

Generate ONLY the Python code, no explanations."""

        if previous_error:
            base_prompt += f"""

PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}

Fix the error and regenerate the complete code."""

        response = self.model.generate_content(base_prompt)
        code = self._extract_code(response.text)

        self._add_to_history("assistant", f"Generated code:\n{code}")

        print("✅ Code generated\n")
        return code

    def run_agent(self, target: str) -> bool:
        """
        Main agent loop: Plan → Generate → Test → Self-correct.

        Args:
            target: Bank name (e.g., 'icici')

        Returns:
            True if parser was successfully generated and tested
        """
        print(f"\n🤖 Starting agent for {target.upper()} bank parser...\n")

        # Setup paths
        pdf_path = f"data/{target}/{target}_sample.pdf"
        csv_path = f"data/{target}/{target}_sample.csv"
        parser_path = f"custom_parsers/{target}_parser.py"
        test_path = f"tests/test_{target}_parser.py"

        # Validate input files exist
        if not Path(pdf_path).exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        if not Path(csv_path).exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False

        # Phase 1: Plan
        plan = self.plan(target, pdf_path, csv_path)

        # Phase 2-4: Generate → Test → Self-correct loop
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🔄 Attempt {attempt}/{self.max_attempts}")

            # Generate code
            if attempt == 1:
                code = self.generate_code(target, pdf_path, csv_path)
            else:
                print(f"🔧 Self-correcting based on previous error...")
                code = self.generate_code(target, pdf_path, csv_path,
                                          previous_error=last_error)

            # Write parser file
            if not self._write_file(parser_path, code):
                continue

            print(f"✅ Parser written to {parser_path}")

            # Generate test file
            test_code = self._generate_test_code(target, parser_path, csv_path)
            if not self._write_file(test_path, test_code):
                continue

            print(f"✅ Test written to {test_path}")

            # Run tests
            print("🧪 Running tests...")
            success, output = self._run_tests(test_path)

            if success:
                print(f"\n✅ SUCCESS! Parser works correctly!")
                print(f"📁 Parser: {parser_path}")
                print(f"📁 Test: {test_path}")
                return True
            else:
                print(f"❌ Tests failed:\n{output[:500]}")
                last_error = output

                if attempt < self.max_attempts:
                    print(f"♻️  Will retry with corrections...")
                else:
                    print(f"\n❌ Max attempts reached. Manual intervention needed.")
                    print(f"Last error:\n{output}")

        return False

    def _generate_test_code(self, target: str, parser_path: str,
                            csv_path: str) -> str:
        """Generate pytest test code for the parser."""
        return f"""import pytest
import pandas as pd
from pathlib import Path
import sys

# Add custom_parsers to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_parsers.{target}_parser import parse


def test_{target}_parser():
    \"\"\"Test that parser output matches expected CSV.\"\"\"
    # Parse PDF
    pdf_path = "data/{target}/{target}_sample.pdf"
    result_df = parse(pdf_path)

    # Load expected CSV
    expected_df = pd.read_csv("data/{target}/{target}_sample.csv")

    # Normalize data types
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        if col in expected_df.columns:
            expected_df[col] = pd.to_numeric(expected_df[col], errors='coerce')

    # Check shape
    assert result_df.shape == expected_df.shape, \\
        f"Shape mismatch: got {{result_df.shape}}, expected {{expected_df.shape}}"

    # Check columns
    assert list(result_df.columns) == list(expected_df.columns), \\
        f"Column mismatch: got {{list(result_df.columns)}}"

    # Check data equality (allowing for minor floating point differences)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), 
        expected_df.reset_index(drop=True),
        check_dtype=False,
        atol=0.01
    )

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_{target}_parser()
"""


def main():
    """CLI entry point."""
    # Load environment variables from .env file
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Autonomous agent for generating bank statement parsers"
    )
    parser.add_argument(
        '--target',
        required=True,
        help='Target bank name (e.g., icici, sbi)'
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('GEMINI_API_KEY'),
        help='Google Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        help='Maximum self-correction attempts (default: 3)'
    )

    args = parser.parse_args()

    if not args.api_key:
        print("❌ Error: No API key provided. Set GEMINI_API_KEY env var or use --api-key")
        sys.exit(1)

    # Create and run agent
    agent = ParserAgent(api_key=args.api_key, max_attempts=args.max_attempts)
    success = agent.run_agent(args.target)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()