#!/usr/bin/env python3
"""
Script to merge track features from track_features_transformed.csv with train_data.csv and test_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_track_id_from_uri(spotify_uri):
    """
    Extract track ID from Spotify URI format (spotify:track:TRACK_ID)
    """
    if pd.isna(spotify_uri) or spotify_uri == '':
        return None
    
    if isinstance(spotify_uri, str) and spotify_uri.startswith('spotify:track:'):
        return spotify_uri.replace('spotify:track:', '')
    
    return None

def main():
    print("Loading train data...")
    # Load train data
    train_data_path = Path("competition_data/train_data.csv")
    train_df = pd.read_csv(train_data_path)
    print(f"Loaded {len(train_df)} rows from train_data.csv")

    print("Loading test data...")
    # Load train data
    test_data_path = Path("competition_data/test_data.csv")
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_df)} rows from test_data.csv")
    
    print("Loading track features...")
    # Load track features
    features_path = Path("track_features_transformed.csv")
    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df)} rows from track_features_transformed.csv")
    
    print("Extracting track IDs from Spotify URIs...")
    # Extract track ID from spotify_track_uri column for both train and test data
    train_df['track_id'] = train_df['spotify_track_uri'].apply(extract_track_id_from_uri)
    test_df['track_id'] = test_df['spotify_track_uri'].apply(extract_track_id_from_uri)
    
    # Count how many tracks have valid track IDs
    valid_track_ids_train = train_df['track_id'].notna().sum()
    valid_track_ids_test = test_df['track_id'].notna().sum()
    print(f"Found {valid_track_ids_train} valid track IDs in train data")
    print(f"Found {valid_track_ids_test} valid track IDs in test data")
    
    print("Merging datasets...")
    # Merge train data with track features on track_id
    # Use left join to keep all train data rows, even if features are missing
    merged_train_df = train_df.merge(
        features_df, 
        on='track_id', 
        how='left',
        suffixes=('', '_feature')
    )

    merged_test_df = test_df.merge(
        features_df, 
        on='track_id', 
        how='left',
        suffixes=('', '_feature')
    )
    
    print(f"Merged train dataset has {len(merged_train_df)} rows")
    print(f"Merged test dataset has {len(merged_test_df)} rows")
    
    # Check how many tracks got features for both datasets
    tracks_with_features_train = merged_train_df['duration_normalized'].notna().sum()
    tracks_without_features_train = merged_train_df['duration_normalized'].isna().sum()
    
    tracks_with_features_test = merged_test_df['duration_normalized'].notna().sum()
    tracks_without_features_test = merged_test_df['duration_normalized'].isna().sum()
    
    print(f"Train data - Tracks with features: {tracks_with_features_train}")
    print(f"Train data - Tracks without features: {tracks_without_features_train}")
    print(f"Test data - Tracks with features: {tracks_with_features_test}")
    print(f"Test data - Tracks without features: {tracks_without_features_test}")
    
    # Remove the temporary track_id column we added
    merged_train_df = merged_train_df.drop('track_id', axis=1)
    merged_test_df = merged_test_df.drop('track_id', axis=1)
    
    print("Saving merged data...")
    # Save the merged datasets
    train_output_path = Path("train_data_with_features.csv")
    test_output_path = Path("test_data_with_features.csv")
    
    merged_train_df.to_csv(train_output_path, index=False)
    merged_test_df.to_csv(test_output_path, index=False)
    
    print(f"Successfully saved merged train data to {train_output_path}")
    print(f"Successfully saved merged test data to {test_output_path}")
    print(f"Final train dataset shape: {merged_train_df.shape}")
    print(f"Final test dataset shape: {merged_test_df.shape}")
    
    # Display column information
    print("\nColumns in merged datasets:")
    for i, col in enumerate(merged_train_df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Show sample of merged data
    print("\nSample of merged train data:")
    print(merged_train_df.head())
    
    print("\nSample of merged test data:")
    print(merged_test_df.head())

if __name__ == "__main__":
    main()
