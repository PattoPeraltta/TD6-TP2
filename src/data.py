import json
import os
import pandas as pd
import numpy as np
from typing import Tuple
from src.features import *

def extract_track_features_no_leakage(data_dir: str, train_track_ids: set = None) -> pd.DataFrame:
    """Extract track features WITHOUT using test data for normalization statistics."""
    transformed_features = []
    durations_ms = []  # Store all durations to calculate mean

    # Get all track files
    track_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.json')]

    print(f"Found {len(track_files)} track files to process...")

    # First pass: collect durations ONLY from training tracks
    print("First pass: collecting durations (TRAINING TRACKS ONLY)...")
    for i, filename in enumerate(track_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Collecting durations from file {i+1}/{len(track_files)}: {filename}")

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract track_id and duration_ms
            track_id = data.get('id', '')
            duration_ms = data.get('duration_ms', 0)
            
            # CRITICAL: Only include durations from training tracks
            if duration_ms > 0 and (train_track_ids is None or track_id in train_track_ids):
                durations_ms.append(duration_ms)

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    # Calculate mean duration statistics ONLY from training data
    mean_duration_ms = np.mean(durations_ms) if durations_ms else 0

    # Calculate 99th percentile for duration clipping ONLY from training data
    duration_99th_percentile = np.percentile(durations_ms, 99) if durations_ms else 0

    print(f"Mean duration calculated (TRAINING ONLY): {mean_duration_ms:.0f} ms ({mean_duration_ms/60000:.2f} minutes) (from {len(durations_ms)} training tracks)")
    print(f"99th percentile duration (TRAINING ONLY): {duration_99th_percentile:.0f} ms ({duration_99th_percentile/60000:.2f} minutes)")

    # Second pass: extract features
    print("Second pass: extracting features...")
    for i, filename in enumerate(track_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(track_files)}: {filename}")

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract raw fields
            track_id = data.get('id', '')
            duration_ms = data.get('duration_ms', 0)
            explicit = data.get('explicit', False)
            popularity = data.get('popularity', 0)
            track_number = data.get('track_number', 0)
            available_markets = data.get('available_markets', [])

            # Extract album_type and release_date from album
            album = data.get('album', {})
            album_type = album.get('album_type', '')
            release_date = album.get('release_date', '')

            # Transform fields
            # Handle zero duration by using mean
            if duration_ms == 0:
                duration_ms = mean_duration_ms

            # Apply clipping at 99th percentile
            clipped_duration_ms = min(duration_ms, duration_99th_percentile)

            # Normalize duration (0-1 scale) using clipped value
            normalized_duration = clipped_duration_ms / duration_99th_percentile if duration_99th_percentile > 0 else 0

            duration_min = duration_ms / 60000.0  # Convert ms to minutes (keep original for raw data)
            available_markets_count = len(available_markets) if available_markets else 0

            # Calculate song release year
            song_release_year = np.nan  # Default to NaN
            if release_date and release_date != "0000":  # Skip invalid dates
                rd = pd.to_datetime(pd.Series(release_date), errors="coerce", format="mixed").iloc[0]
                song_release_year = rd.year if pd.notna(rd) and rd.year > 0 else np.nan

            # Store transformed features
            transformed_data = {
                'track_id': track_id,
                'duration_normalized': normalized_duration,
                'explicit': explicit,
                'popularity': popularity,
                'track_number': track_number,
                'available_markets_count': available_markets_count,
                'album_type': album_type,
                'release_year': song_release_year
            }
            transformed_features.append(transformed_data)

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Convert to DataFrame first, then apply transformations
    transformed_features_df = pd.DataFrame(transformed_features)
    
    # Apply transformations to the DataFrame
    transformed_features_df["explicit"] = transformed_features_df["explicit"].astype(int)
    for c in ["popularity", "track_number", "available_markets_count", "release_year", "duration_normalized"]:
        transformed_features_df[c] = pd.to_numeric(transformed_features_df[c], errors="coerce").astype("float32")

    print(f"Successfully processed {len(transformed_features_df)} tracks")

    return transformed_features_df

def load_and_merge_track_features(train_path: str, test_path: str, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path,  sep='\t')

    # Extract track_id from spotify_track_uri
    train_df['track_id'] = train_df['spotify_track_uri'].str.replace('spotify:track:', '')
    test_df['track_id'] = test_df['spotify_track_uri'].str.replace('spotify:track:', '')

    n_train = len(train_df)
    train_df = train_df.merge(features, on='track_id', how='left')
    test_df = test_df.merge(features, on='track_id',  how='left')

    cov = train_df["duration_normalized"].notna().mean()
    print(f"[merge] feature coverage on train: {cov:.1%} ({train_df['duration_normalized'].notna().sum()}/{n_train})")

    boolean_columns = ['explicit', 'shuffle', 'offline', 'incognito_mode']
    for col in boolean_columns:
        if col in train_df.columns:
            # Fill NaN values with 0 (False) before converting to int
            train_df[col] = train_df[col].fillna(0).astype("int8")
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0).astype("int8")

    categorical_columns = ['platform', 'conn_country', 'album_type', 'username']
    for col in categorical_columns:
        if col in train_df.columns:
            # Fill NaN values with 'unknown' before converting to category
            train_df[col] = train_df[col].fillna('unknown').astype('category')
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna('unknown').astype('category')

    # Cambiamos la columna reason_end a 1 (fwdbtn) o 0 (otro)
    train_df['reason_end'] = train_df['reason_end'].apply(lambda x: 1 if x == 'fwdbtn' else 0)
    train_df.rename(columns={'reason_end': 'fwdbtn'}, inplace=True)

    # Move the 'fwdbtn' column to the end
    fwdbtn_col = train_df.pop('fwdbtn')
    train_df['fwdbtn'] = fwdbtn_col

    train_df.to_csv("competition_data/raw_with_spotify_api_data/train_data_raw_with_spotify_api_data.csv")
    test_df.to_csv("competition_data/raw_with_spotify_api_data/test_data_raw_with_spotify_api_data.csv")

    return train_df, test_df

def per_user_time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    For each user, sort by temporal features (year, month, day) and take the first train_frac as train, the rest as val.
    Returns train/val indices (np.ndarray).
    """
    # Check if we have temporal features available
    temporal_cols = ['year', 'month', 'day']
    if not all(col in df.columns for col in temporal_cols):
        raise ValueError(f"Missing temporal columns {temporal_cols}. Make sure add_temporal() has been applied.")
    
    # Create a composite timestamp for sorting
    dfi = df[['username'] + temporal_cols].copy()
    
    # Create a sortable datetime column from year, month, day
    dfi['composite_ts'] = pd.to_datetime(dfi[['year', 'month', 'day']], errors='coerce')
    
    if dfi['composite_ts'].isna().any():
        raise ValueError("Found invalid dates in temporal features. Clean data first.")

    # rank within user based on composite timestamp
    dfi = dfi.sort_values(['username', 'composite_ts']).reset_index()
    dfi["rank"] = dfi.groupby('username').cumcount()
    dfi["n_user"] = dfi.groupby('username')['username'].transform("size")
    # cutoff per user (floor since cumcount starts at 0)
    dfi["cut"] = np.floor(dfi["n_user"] * train_frac).astype(int).clip(lower=1)

    # train if rank < cut, else val
    train_mask = dfi["rank"] < dfi["cut"]
    train_idx = dfi.loc[train_mask, "index"].to_numpy()
    val_idx = dfi.loc[~train_mask, "index"].to_numpy()

    return train_idx, val_idx

def make_data_with_features(train_path: str, test_path: str, track_data_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Unified function that loads data, merges Spotify features, applies feature engineering,
    and returns train/validation/test splits ready for modeling.
    """
    print("🔄 Loading and merging Spotify features...")

    try:
        train_df = pd.read_csv("competition_data/raw_with_spotify_api_data/train_data_raw_with_spotify_api_data.csv")
        test_df = pd.read_csv("competition_data/raw_with_spotify_api_data/test_data_raw_with_spotify_api_data.csv")
        print("Found existing raw files with Spotify data!")
    except FileNotFoundError:
        print("Raw files with Spotify data not found, creating them...")
        
        # Load raw data to get training track IDs
        train_df_raw = pd.read_csv(train_path, sep='\t')
        test_df_raw = pd.read_csv(test_path, sep='\t')
        
        # Extract track IDs from training data only
        train_df_raw['track_id'] = train_df_raw['spotify_track_uri'].str.replace('spotify:track:', '')
        train_track_ids = set(train_df_raw['track_id'].unique())
        
        print(f"Found {len(train_track_ids)} unique track IDs in training data")
        
        # Extract features WITHOUT test data leakage
        features = extract_track_features_no_leakage(track_data_path, train_track_ids)
        train_df, test_df = load_and_merge_track_features(train_path, test_path, features)

    print("🎯 Applying feature engineering...")
    # Apply feature engineering WITHOUT username OHE first (to keep username for temporal split)
    train_df_temp = make_features(train_df.copy(), include_username_ohe=False)
    
    print("📊 Creating train/validation split...")
    # Create temporal split using username column (before OHE)
    train_idx, val_idx = per_user_time_split(train_df_temp, train_frac=0.8)
    
    print("🔧 Adding username one-hot encoding to final features...")
    # Now apply full feature engineering including username OHE
    train_df_processed = make_features(train_df.copy(), include_username_ohe=True)
    test_df_processed = make_features(test_df.copy(), include_username_ohe=True)

    # Define feature columns (exclude target and ID columns)
    exclude_cols = ['reason_end', 'obs_id', 'username', 'ts', 'offline_timestamp', 'fwdbtn']
    feature_cols = [col for col in train_df_processed.columns if col not in exclude_cols]
    
    # Handle missing values in features WITHOUT test data leakage
    print("🔧 Imputing missing values (NO LEAKAGE)...")
    for col in feature_cols:
        if train_df_processed[col].dtype == 'object' or train_df_processed[col].dtype == 'category':
            # Fill categorical columns with most frequent category from training data
            most_frequent = train_df_processed[col].mode()[0] if not train_df_processed[col].mode().empty else 'unknown'
            train_df_processed[col] = train_df_processed[col].fillna(most_frequent)
            test_df_processed[col] = test_df_processed[col].fillna(most_frequent)
        else:
            # Fill numeric columns with median from TRAINING DATA ONLY
            median_val = train_df_processed[col].median()
            train_df_processed[col] = train_df_processed[col].fillna(median_val)
            test_df_processed[col] = test_df_processed[col].fillna(median_val)
            print(f"   Imputed {col} with training median: {median_val:.4f}")

    # Create final splits
    X_train = train_df_processed.iloc[train_idx][feature_cols]
    y_train = train_df_processed.iloc[train_idx]['fwdbtn']
    X_val = train_df_processed.iloc[val_idx][feature_cols]
    y_val = train_df_processed.iloc[val_idx]['fwdbtn']
    X_test = test_df_processed[feature_cols]

    print(f"✅ Data processing completed!")
    print(f"   Training features: {X_train.shape}")
    print(f"   Validation features: {X_val.shape}")
    print(f"   Test features: {X_test.shape}")
    print(f"   Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"   Validation target distribution: {y_val.value_counts().to_dict()}")
    print(f"   Feature columns: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test