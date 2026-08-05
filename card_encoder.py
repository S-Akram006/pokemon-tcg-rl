import pandas as pd
import json

DATA_PATH = "data/EN_Card_Data.csv"
OUTPUT_PATH = "data/card_lookup.json"

TYPE_MAP = {
    '{G}': 1, '{R}': 2, '{W}': 3, '{L}': 4,
    '{P}': 5, '{F}': 6, '{D}': 7, '{M}': 8,
    '{C}': 9, '{N}': 10
}

def infer_category(stage_val):
    if pd.isna(stage_val):
        return 0
    stage_str = str(stage_val).lower()
    if 'energy' in stage_str:
        return 3  # Energy
    elif any(k in stage_str for k in ['item', 'supporter', 'stadium', 'tool']):
        return 2  # Trainer
    elif 'pokémon' in stage_str or 'pokemon' in stage_str:
        return 1  # Pokémon
    return 0

def clean_numeric(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace('+', '').replace('×', '').strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def generate_lookup():
    df = pd.read_csv(DATA_PATH)
    stage_col = 'Stage (Pokémon)/Type (Energy and Trainer)'
    
    df['category_code'] = df[stage_col].apply(infer_category)
    df['type_code'] = df['Type'].map(TYPE_MAP).fillna(0)
    df['hp_clean'] = df['HP'].apply(clean_numeric)
    df['retreat_clean'] = df['Retreat'].apply(clean_numeric)
    df['damage_clean'] = df['Damage'].apply(clean_numeric)
    
    # Create lookup dictionary mapping Card ID -> Feature Vector
    card_lookup = {}
    for _, row in df.iterrows():
        card_id = int(row['Card ID'])
        card_lookup[card_id] = {
            "name": str(row['Card Name']),
            "features": [
                float(row['category_code']),
                float(row['type_code']),
                float(row['hp_clean']),
                float(row['retreat_clean']),
                float(row['damage_clean'])
            ]
        }
        
    # Save to JSON
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(card_lookup, f, indent=2)
        
    print(f"=== LOOKUP DICTIONARY SAVED TO {OUTPUT_PATH} ===")
    print(f"Total Unique Cards Processed: {len(card_lookup)}")

if __name__ == "__main__":
    generate_lookup()