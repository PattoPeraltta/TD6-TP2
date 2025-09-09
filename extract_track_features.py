#!/usr/bin/env python3
"""
Script to extract and transform track features from Spotify API data files.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def extract_track_features(data_dir: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract track features from all track JSON files in the specified directory.
    
    Args:
        data_dir: Path to directory containing Spotify API data files
        
    Returns:
        Tuple of (raw_features, transformed_features) lists
    """
    raw_features = []
    transformed_features = []
    release_dates = []  # Store all release dates to calculate mean
    durations_ms = []  # Store all durations to calculate mean
    
    # Get all track files
    track_files = [f for f in os.listdir(data_dir) if f.startswith('spotify:track:') and f.endswith('.json')]
    
    print(f"Found {len(track_files)} track files to process...")
    
    # First pass: collect all release dates and durations
    print("First pass: collecting release dates and durations...")
    for i, filename in enumerate(track_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Collecting dates from file {i+1}/{len(track_files)}: {filename}")
            
        file_path = os.path.join(data_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract duration_ms
            duration_ms = data.get('duration_ms', 0)
            if duration_ms > 0:  # Only include valid durations
                durations_ms.append(duration_ms)
            
            # Extract release_date from album
            album = data.get('album', {})
            release_date = album.get('release_date', '')
            
            if release_date and release_date != "0000":  # Skip invalid dates
                try:
                    # Handle different date formats
                    if len(release_date) == 4:  # Year only
                        release_year = int(release_date)
                        if release_year > 0:  # Only include valid years
                            current_year = datetime.now().year
                            song_age = current_year - release_year
                            release_dates.append(song_age)
                    elif len(release_date) >= 7:  # Year-month or full date
                        release_dt = datetime.strptime(release_date[:10], '%Y-%m-%d')
                        current_dt = datetime.now()
                        song_age = (current_dt - release_dt).days / 365.25  # More accurate age in years
                        if song_age >= 0:  # Only include non-negative ages
                            release_dates.append(song_age)
                except (ValueError, TypeError):
                    continue
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue
    
    # Calculate mean release date age and duration statistics
    mean_song_age = sum(release_dates) / len(release_dates) if release_dates else 0
    mean_duration_ms = sum(durations_ms) / len(durations_ms) if durations_ms else 0
    
    # Calculate 99th percentile for duration clipping
    duration_99th_percentile = np.percentile(durations_ms, 99) if durations_ms else 0
    
    print(f"Mean song age calculated: {mean_song_age:.2f} years (from {len(release_dates)} tracks with release dates)")
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
            
            # Extract artist_uri (first artist)
            artists = data.get('artists', [])
            artist_uri = artists[0].get('uri', '') if artists else ''
            
            # Extract album_type and release_date from album
            album = data.get('album', {})
            album_type = album.get('album_type', '')
            release_date = album.get('release_date', '')
            
            # Store raw features
            raw_data = {
                'track_id': track_id,
                'duration_ms': duration_ms,
                'explicit': explicit,
                'popularity': popularity,
                'track_number': track_number,
                'available_markets': available_markets,
                'artist_uri': artist_uri,
                'album_type': album_type,
                'release_date': release_date
            }
            raw_features.append(raw_data)
            
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
            
            # Calculate song age in years
            song_age = mean_song_age  # Default to mean age
            if release_date and release_date != "0000":  # Skip invalid dates
                try:
                    # Handle different date formats
                    if len(release_date) == 4:  # Year only
                        release_year = int(release_date)
                        if release_year > 0:  # Only include valid years
                            current_year = datetime.now().year
                            song_age = current_year - release_year
                    elif len(release_date) >= 7:  # Year-month or full date
                        release_dt = datetime.strptime(release_date[:10], '%Y-%m-%d')
                        current_dt = datetime.now()
                        song_age = (current_dt - release_dt).days / 365.25  # More accurate age in years
                        if song_age < 0:  # If future date, use mean
                            song_age = mean_song_age
                except (ValueError, TypeError):
                    song_age = mean_song_age  # Use mean if parsing fails
            
            # Store transformed features
            transformed_data = {
                'track_id': track_id,
                'duration_normalized': normalized_duration,
                'explicit': explicit,
                'popularity': popularity,
                'track_number': track_number,
                'available_markets_count': available_markets_count,
                'artist_uri': artist_uri,
                'album_type': album_type,
                'song_age': song_age
            }
            transformed_features.append(transformed_data)
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Successfully processed {len(raw_features)} tracks")
    return raw_features, transformed_features

def main():
    """Main function to extract track features and save to CSV."""
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'competition_data', 'spotify_api_data')
    raw_output_file = os.path.join(script_dir, 'track_features_raw.csv')
    transformed_output_file = os.path.join(script_dir, 'track_features_transformed.csv')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    print(f"Extracting track features from: {data_dir}")
    print(f"Raw data will be saved to: {raw_output_file}")
    print(f"Transformed data will be saved to: {transformed_output_file}")
    
    # Extract features
    raw_features, transformed_features = extract_track_features(data_dir)
    
    if not raw_features or not transformed_features:
        print("No track features extracted. Exiting.")
        return
    
    # Process raw features
    print(f"\nProcessing raw features...")
    raw_df = pd.DataFrame(raw_features)
    
    # Reorder columns for better readability
    raw_column_order = [
        'track_id', 'duration_ms', 'explicit', 'popularity', 
        'track_number', 'available_markets', 'artist_uri', 
        'album_type', 'release_date'
    ]
    raw_df = raw_df[raw_column_order]
    
    # Save raw data to CSV
    raw_df.to_csv(raw_output_file, index=False)
    
    # Process transformed features
    print(f"Processing transformed features...")
    transformed_df = pd.DataFrame(transformed_features)
    
    # Reorder columns for better readability
    transformed_column_order = [
        'track_id', 'duration_normalized', 'explicit', 'popularity', 
        'track_number', 'available_markets_count', 'artist_uri', 
        'album_type', 'song_age'
    ]
    transformed_df = transformed_df[transformed_column_order]
    
    # Save transformed data to CSV
    transformed_df.to_csv(transformed_output_file, index=False)
    
    print(f"\nFiles saved successfully!")
    print(f"Raw features: {raw_output_file}")
    print(f"Transformed features: {transformed_output_file}")
    print(f"Total tracks processed: {len(raw_df)}")
    
    print(f"\nRaw data columns: {list(raw_df.columns)}")
    print(f"Transformed data columns: {list(transformed_df.columns)}")
    
    # Display basic statistics for raw data
    print(f"\nRaw data statistics:")
    print(f"Duration range: {raw_df['duration_ms'].min()} - {raw_df['duration_ms'].max()} ms")
    print(f"Popularity range: {raw_df['popularity'].min()} - {raw_df['popularity'].max()}")
    print(f"Track number range: {raw_df['track_number'].min()} - {raw_df['track_number'].max()}")
    
    # Display basic statistics for transformed data
    print(f"\nTransformed data statistics:")
    print(f"Normalized duration range: {transformed_df['duration_normalized'].min():.4f} - {transformed_df['duration_normalized'].max():.4f}")
    print(f"Popularity range: {transformed_df['popularity'].min()} - {transformed_df['popularity'].max()}")
    print(f"Song age range: {transformed_df['song_age'].min():.1f} - {transformed_df['song_age'].max():.1f} years")
    print(f"Available markets count range: {transformed_df['available_markets_count'].min()} - {transformed_df['available_markets_count'].max()}")

if __name__ == "__main__":
    main()
