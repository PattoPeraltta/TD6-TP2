import json
import os
import pandas as pd
import numpy as np
from typing import Tuple

def extract_track_features(data_dir: str) -> pd.DataFrame:
    transformed_features = []
    durations_ms = []  # Store all durations to calculate mean

    # Get all track files
    track_files = [f for f in os.listdir(data_dir) if f.startswith('spotify:track:') and f.endswith('.json')]

    print(f"Found {len(track_files)} track files to process...")

    # First pass: collect durations
    print("First pass: collecting durations...")
    for i, filename in enumerate(track_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Collecting durations from file {i+1}/{len(track_files)}: {filename}")

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract duration_ms
            duration_ms = data.get('duration_ms', 0)
            if duration_ms > 0:  # Only include valid durations
                durations_ms.append(duration_ms)

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    # Calculate mean duration statistics
    mean_duration_ms = sum(durations_ms) / len(durations_ms) if durations_ms else 0

    # Calculate 99th percentile for duration clipping
    duration_99th_percentile = np.percentile(durations_ms, 99) if durations_ms else 0

    print(f"Mean duration calculated: {mean_duration_ms:.0f} ms ({mean_duration_ms/60000:.2f} minutes) (from {len(durations_ms)} tracks with valid durations)")
    print(f"99th percentile duration: {duration_99th_percentile:.0f} ms ({duration_99th_percentile/60000:.2f} minutes)")

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

    transformed_features["explicit"] = transformed_features["explicit"].astype(int)
    for c in ["popularity", "track_number", "available_markets_count", "release_year", "duration_normalized"]:
        transformed_features[c] = pd.to_numeric(transformed_features[c], errors="coerce").astype("float32")

    print(f"Successfully processed {len(transformed_features)} tracks")
    transformed_features_df = pd.DataFrame(transformed_features)

    return transformed_features_df


def load_and_merge_track_features(train_path: str, test_path: str, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path,  sep='\t')

    n_train = len(train_df)
    train_df = train_df.merge(features, on='track_id', how='left')
    test_df = test_df.merge(features, on='track_id',  how='left')

    cov = train_df["duration_normalized"].notna().mean()
    print(f"[merge] feature coverage on train: {cov:.1%} ({train_df['duration_normalized'].notna().sum()}/{n_train})")

    boolean_columns = ['explicit', 'shuffle', 'offline', 'incognito_mode']
    for col in boolean_columns:
        train_df[col] = train_df[col].astype("int8")
        test_df[col] = test_df[col].astype("int8")

    categorical_columns = ['platform', 'conn_country', 'album_type', 'username']
    for col in categorical_columns:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    # Cambiamos la columna reason_end a 1 (fwdbtn) o 0 (otro)
    train_df['reason_end'] = train_df['reason_end'].apply(lambda x: 1 if x == 'fwdbtn' else 0)
    train_df.rename(columns={'reason_end': 'fwdbtn'}, inplace=True)

    # Move the 'fwdbtn' column to the end
    fwdbtn_col = train_df.pop('fwdbtn')
    train_df['fwdbtn'] = fwdbtn_col

    return train_df, test_df


def per_user_time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    For each user, sort by timestamp and take the first train_frac as train, the rest as val.
    Returns train/val indices (np.ndarray).
    """
    dfi = df[['username', 'ts']].copy()
    dfi['ts'] = pd.to_datetime(dfi['ts'], utc=True, errors="coerce")
    if dfi['ts'].isna().any():
        raise ValueError(f"Found NaT in {'ts'}; clean timestamps first.")

    # rank within user
    dfi = dfi.sort_values(['username', 'ts']).reset_index()
    dfi["rank"] = dfi.groupby('username').cumcount()
    dfi["n_user"] = dfi.groupby('username')['username'].transform("size")
    # cutoff per user (floor since cumcount starts at 0)
    dfi["cut"] = np.floor(dfi["n_user"] * train_frac).astype(int).clip(lower=1)

    # train if rank < cut, else val
    train_mask = dfi["rank"] < dfi["cut"]
    train_idx = dfi.loc[train_mask, "index"].to_numpy()
    val_idx = dfi.loc[~train_mask, "index"].to_numpy()

    return train_idx, val_idx

def make_data(train_path: str, test_path: str, track_data_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    features = extract_track_features(track_data_path)
    train_df, test_df = load_and_merge_track_features(train_path, test_path, features)
    train_idx, val_idx = per_user_time_split(train_df, train_frac=0.8)

    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    feature_cols = [c for c in common_cols if c != 'fwdbtn']

    X_train = train_df.iloc[train_idx][feature_cols]
    y_train = train_df.iloc[train_idx]['fwdbtn']
    X_val = train_df.iloc[val_idx][feature_cols]
    y_val = train_df.iloc[val_idx]['fwdbtn']
    X_test = test_df[feature_cols]

    return X_train, y_train, X_val, y_val, X_test