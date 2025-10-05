import pandas as pd
import tabula
import numpy as np
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses an ICICI bank statement PDF and returns a pandas DataFrame
    with transaction details.

    Args:
        pdf_path (str): The path to the ICICI bank statement PDF file.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
                      ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].
                      'Date' will be in 'DD-MM-YYYY' string format.
                      'Debit Amt', 'Credit Amt', 'Balance' will be float type.

    Raises:
        ValueError: If no tables are found in the PDF or parsing fails.
        FileNotFoundError: If the specified PDF path does not exist.
        Exception: For other unexpected errors during processing.
    """
    target_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    all_extracted_dfs = []

    try:
        # Extract tables from PDF.
        # pages='all' extracts from all pages.
        # multiple_tables=False tries to extract a single main table per page.
        # stream=True is often better for structured tables, guess=True helps identify table areas.
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=False, stream=True, guess=True)
        
        if not tables:
            raise ValueError("No tables found in the PDF. Ensure it's a valid ICICI statement or that tabula can extract tables.")
        
        for df_page in tables:
            if df_page.empty:
                continue

            # Drop rows and columns that are entirely NaN
            df_page = df_page.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if df_page.empty:
                continue
            
            # Normalize potential column names for easier searching
            # This handles multi-level headers created by tabula and cleans strings
            df_page.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df_page.columns.values]
            df_page.columns = [re.sub(r'[^a-zA-Z0-9]+', ' ', col).strip().lower() for col in df_page.columns]
            
            # Find the actual header row within the dataframe by looking for keywords
            header_row_idx = -1
            
            # Iterate through potential header rows (first few rows)
            for r_idx, row in df_page.head(10).iterrows(): # Check first 10 rows for header
                row_str = ' '.join(row.dropna().astype(str).values).lower()
                
                # A row is a strong candidate for header if it contains multiple identifying keywords
                date_found = any(k in row_str for k in ['date', 'txn date', 'transaction date'])
                desc_found = any(k in row_str for k in ['description', 'particulars', 'narration'])
                amt_found = any(k in row_str for k in ['debit', 'withdrawal', 'credit', 'deposit', 'balance'])

                if date_found and desc_found and amt_found:
                    header_row_idx = r_idx
                    break
            
            if header_row_idx != -1:
                # Use the identified row's values as new column names
                # Slice the dataframe to get the header row and the data below it
                header_values = df_page.iloc[header_row_idx].values
                # Clean these header values before assigning
                cleaned_header_values = [re.sub(r'[^a-zA-Z0-9]+', ' ', str(val)).strip().lower() for val in header_values]
                
                # Assign new columns, handling potential duplicates by appending index
                unique_cols = {}
                final_header = []
                for h in cleaned_header_values:
                    if h in unique_cols:
                        unique_cols[h] += 1
                        final_header.append(f"{h}_{unique_cols[h]}")
                    else:
                        unique_cols[h] = 0
                        final_header.append(h)

                df_page = df_page.iloc[header_row_idx + 1:].copy()
                df_page.columns = final_header[:len(df_page.columns)] # Trim header if too long
                df_page.reset_index(drop=True, inplace=True)
            
            # Remove empty string column names that might arise from cleaning headers
            df_page = df_page.loc[:, df_page.columns.notna() & (df_page.columns != '')]

            # Now, map the current DataFrame columns to our target_columns based on keywords
            col_mapping = {}
            # Prioritize more specific names
            if 'date' in df_page.columns: col_mapping['date'] = 'Date'
            elif 'transaction date' in df_page.columns: col_mapping['transaction date'] = 'Date'
            
            if 'description' in df_page.columns: col_mapping['description'] = 'Description'
            elif 'particulars' in df_page.columns: col_mapping['particulars'] = 'Description'
            elif 'narration' in df_page.columns: col_mapping['narration'] = 'Description'
            
            if 'debit' in df_page.columns: col_mapping['debit'] = 'Debit Amt'
            elif 'withdrawal' in df_page.columns: col_mapping['withdrawal'] = 'Debit Amt'

            if 'credit' in df_page.columns: col_mapping['credit'] = 'Credit Amt'
            elif 'deposit' in df_page.columns: col_mapping['deposit'] = 'Credit Amt'

            if 'balance' in df_page.columns: col_mapping['balance'] = 'Balance'
            elif 'closing balance' in df_page.columns: col_mapping['closing balance'] = 'Balance'
            
            # Rename columns
            df_page = df_page.rename(columns=col_mapping)

            # Ensure only relevant columns are kept and in order. Fill missing with NaN.
            processed_df_page = pd.DataFrame(columns=target_columns)
            for col in target_columns:
                if col in df_page.columns:
                    processed_df_page[col] = df_page[col]
                else:
                    processed_df_page[col] = np.nan # Placeholder for missing columns

            all_extracted_dfs.append(processed_df_page)

    except FileNotFoundError:
        raise FileNotFoundError(f"The PDF file was not found at: {pdf_path}")
    except ValueError as e:
        raise ValueError(f"Error parsing PDF: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during PDF processing: {e}")

    if not all_extracted_dfs:
        raise ValueError("No valid transaction data could be extracted from the PDF.")

    df_combined = pd.concat(all_extracted_dfs, ignore_index=True)

    # --- Post-extraction Cleaning and Normalization ---
    
    # Drop rows that are entirely NaN after initial concatenation
    df_combined.dropna(how='all', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    # 1. Convert 'Date' column to datetime objects first to identify valid dates
    # Invalid dates will become NaT (Not a Time)
    df_combined['Date_Parsed'] = pd.to_datetime(df_combined['Date'], format='%d-%m-%Y', errors='coerce')

    # 2. Handle multi-line descriptions and filter out non-transaction rows
    final_transactions = []
    current_transaction = None

    for idx, row in df_combined.iterrows():
        # Tentatively parse debit/credit to check for meaningful values
        temp_debit = pd.to_numeric(str(row['Debit Amt']).replace(',', '').strip(), errors='coerce')
        temp_credit = pd.to_numeric(str(row['Credit Amt']).replace(',', '').strip(), errors='coerce')
        
        # A row is considered a new transaction if it has a valid date, non-empty description,
        # AND a non-zero debit or credit amount. This filters out summary/opening balance rows.
        is_transaction_start = pd.notna(row['Date_Parsed']) and \
                               (pd.notna(row['Description']) and str(row['Description']).strip() != '') and \
                               ((pd.notna(temp_debit) and temp_debit > 0) or \
                                (pd.notna(temp_credit) and temp_credit > 0))

        if is_transaction_start:
            if current_transaction is not None:
                final_transactions.append(current_transaction)
            current_transaction = row.copy()
            current_transaction['Description'] = str(current_transaction['Description']).strip()
        else:
            # If not a transaction start, it could be a continuation of the description,
            # or simply junk/summary data without valid transaction fields.
            if current_transaction is not None and pd.notna(row['Description']) and str(row['Description']).strip() != '':
                current_transaction['Description'] += ' ' + str(row['Description']).strip()
            # If no active transaction or no description content, this row is likely noise and will be discarded.

    if current_transaction is not None:
        final_transactions.append(current_transaction)

    df_final = pd.DataFrame(final_transactions, columns=df_combined.columns)

    # Drop the temporary 'Date_Parsed' column
    df_final.drop(columns=['Date_Parsed'], inplace=True)

    # Filter out rows where the Date column is still invalid (e.g., rows that only contained junk)
    df_final = df_final[pd.notna(pd.to_datetime(df_final['Date'], format='%d-%m-%Y', errors='coerce'))]


    # 3. Type Conversion and Final Cleaning for Numeric Columns
    numeric_cols = ['Debit Amt', 'Credit Amt', 'Balance']
    for col in numeric_cols:
        # Fill NaN with 0, convert to string, remove non-numeric characters except dot for decimals.
        df_final[col] = df_final[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        # Convert to numeric, coerce errors to NaN (e.g., if it was an empty string), then fill NaN with 0.0
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0.0)

    # 4. Final 'Date' format conversion to 'DD-MM-YYYY' string as required by the test.
    # Convert back to datetime first (to ensure consistency), then format to string.
    df_final['Date'] = pd.to_datetime(df_final['Date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')

    # Ensure Description is string type and strip whitespace
    df_final['Description'] = df_final['Description'].astype(str).str.strip()
    
    # Ensure final DataFrame has exactly the required columns in the correct order
    df_final = df_final[target_columns].copy()

    return df_final