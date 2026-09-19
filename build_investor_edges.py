"""
Phase 1: Entity Disentanglement Script
Parses the static time-series CSV to extract distinct Investor nodes and map their edges.
"""
import pandas as pd
from pathlib import Path

def generate_investor_edges(*args, **kwargs):
    """
    Bulletproof signature: accepts any arguments server.py throws at it.
    """
    base_dir = Path(__file__).parent
    
    # Try to grab the path if server.py passed it, otherwise default
    input_file = base_dir / "enriched_timeseries_market_intel.csv"
    if args and args[0]:
        input_file = Path(args[0])
    elif 'csv_path' in kwargs and kwargs['csv_path']:
        input_file = Path(kwargs['csv_path'])

    output_file = base_dir / "investor_edges.csv"

    if not input_file.exists():
        print(f"❌ Error: {input_file.name} not found.")
        return False

    print(f"⏳ Extracting investor entities from {input_file.name}...")
    
    # Read the data
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return False

    # We only need the latest investor cap table for the base graph.
    df_latest = df.sort_values('Date').drop_duplicates(subset=['Company'], keep='last')
    df_investors = df_latest.dropna(subset=['Investors']).copy()

    # Step 1: Split the comma-separated strings
    df_investors['Investor'] = df_investors['Investors'].astype(str).str.split(',')

    # Step 2: Explode into separate rows
    exploded_df = df_investors.explode('Investor')

    # Step 3: Clean whitespace
    exploded_df['Investor'] = exploded_df['Investor'].str.strip()

    # Step 4: Remove empty artifacts
    exploded_df = exploded_df[exploded_df['Investor'] != '']

    # Keep only target edges
    final_edges = exploded_df[['Investor', 'Company']].drop_duplicates()

    # 🔴 FIX: Rename both 'Investor' and 'Company' columns to match server.py schema
    final_edges = final_edges.rename(columns={
        'Investor': 'Investor_Name',
        'Company': 'Company_Name'
    })

    # Save to CSV
    final_edges.to_csv(output_file, index=False)
    print(f"✅ Successfully mapped {len(final_edges)} distinct investor-company edges!")
    print(f"✅ Saved to {output_file.name}")
    
    return True

if __name__ == "__main__":
    generate_investor_edges()