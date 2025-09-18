import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import hashlib
import pickle

def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all')

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['year'] = df['ts'].dt.year
    df['month'] = df['ts'].dt.month
    df['day'] = df['ts'].dt.day
    df['time_of_day'] = df['ts'].dt.time
    time_seconds = df['time_of_day'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    
    year = df.pop('year')
    month = df.pop('month')
    day = df.pop('day')
    day_of_week = pd.to_datetime({'year': year, 'month': month, 'day': day}).dt.day_name().astype('category')
    is_weekend = ((day_of_week == 'Saturday') | (day_of_week == 'Sunday')).astype("int8")
    is_morning = ((time_seconds >= 21600) & (time_seconds < 43200)).astype("int8")
    is_afternoon = ((time_seconds >= 43200) & (time_seconds < 64800)).astype("int8")
    is_evening = ((time_seconds >= 64800) & (time_seconds < 79200)).astype("int8")
    is_night = (((time_seconds >= 79200) | (time_seconds < 86400)) | (time_seconds < 21600)).astype("int8")

    df.insert(0, 'year', year)
    df.insert(1, 'month', month)
    df.insert(2, 'day', day)
    df.insert(3, 'day_of_week', day_of_week)
    df.insert(4, 'is_weekend', is_weekend)
    df.insert(5, 'is_morning', is_morning)
    df.insert(6, 'is_afternoon', is_afternoon)
    df.insert(7, 'is_evening', is_evening)
    df.insert(8, 'is_night', is_night)
    
    # Cyclical encoding for time_of_day
    # Convert time_of_day to seconds since midnight
    seconds_in_day = 24 * 60 * 60
    df.pop('time_of_day')
    time_of_day_sin = np.sin(2 * np.pi * time_seconds / seconds_in_day)
    time_of_day_cos = np.cos(2 * np.pi * time_seconds / seconds_in_day)
    df.insert(9, 'time_of_day_sin', time_of_day_sin)
    df.insert(10, 'time_of_day_cos', time_of_day_cos)

    return df

def add_track_order_in_day(df: pd.DataFrame) -> pd.DataFrame:
    df["track_order_in_day"] = (df.groupby([df["username"], df["ts"].dt.date])["ts"].rank(method="first").astype(int))
    return df

def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_song'] = df['master_metadata_track_name'].notnull().astype("int8")
    df['is_episode'] = df['episode_name'].notnull().astype("int8")

    df['song_age'] = df['year'] - df['release_year']
    df['song_age'] = df['song_age'].apply(lambda x: x if x >= 0 else 0)
    return df

def map_platform(platform_str: str) -> str:
    if pd.isnull(platform_str):
        return 'other'
    s = platform_str.lower()
    if 'windows' in s:
        return 'windows'
    elif 'android' in s:
        return 'android'
    elif 'ios' in s or 'iphone' in s or 'ipad' in s:
        return 'ios'
    elif 'linux' in s:
        return 'linux'
    elif 'osx' in s or 'mac' in s or 'os x' in s:
        return 'osx'
    elif 'cast' in s or 'chromecast' in s:
        return 'cast'
    elif 'webplayer' in s or 'web player' in s or 'web' in s:
        return 'webplayer'
    else:
        return 'other'

def add_platform(df: pd.DataFrame) -> pd.DataFrame:
    df['platform'] = df['platform'].apply(map_platform)
    return df

def add_per_user_skip_rate(train_df: pd.DataFrame, test_df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Create training subset for calculating user skip rates
    train_subset = train_df.iloc[train_idx]
    
    # Calculamos el user_skip_rate SOLO EN TRAIN
    user_skip_rate = ( train_subset.groupby("username")['fwdbtn']
                        .mean()
                        .rename("per_user_skip_rate")
                    )

    # Add per_user_skip_rate to all dataframes
    train_df = train_df.merge(user_skip_rate, on="username", how="left")
    test_df = test_df.merge(user_skip_rate, on="username", how="left")

    global_skip_rate = train_subset['fwdbtn'].mean()
    for df in [train_df, test_df]:
        df["per_user_skip_rate"] = df["per_user_skip_rate"].fillna(global_skip_rate)

    return train_df, test_df

def add_bin_counting(df: pd.DataFrame, verbose: bool = False, cache_dir: str = "data") -> pd.DataFrame:
    """
    Add bin counting features for user listening activity in different time windows.
    
    For each observation, counts:
    - Total songs played in the last 20min, 1h, 6h, 12h, 24h
    - Total songs skipped in the last 20min, 1h, 6h, 12h, 24h  
    
    Args:
        df: DataFrame with columns ['ts', 'username'] and optionally 'fwdbtn'
        verbose: Whether to print progress information
        cache_dir: Directory to store cached results
        
    Returns:
        DataFrame with additional bin counting features
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename based on data hash
    # Use only available columns for hashing
    hash_columns = ['ts', 'username']
    if 'fwdbtn' in df.columns:
        hash_columns.append('fwdbtn')
    
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df[hash_columns]).values
    ).hexdigest()
    cache_file = os.path.join(cache_dir, f"bin_counting_cache_{data_hash}.pkl")
    
    # Check if cached file exists
    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading bin counting features from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if verbose:
                print("Successfully loaded cached bin counting features")
            return cached_data
        except Exception as e:
            if verbose:
                print(f"Error loading cache file: {e}. Recomputing...")
    
    if verbose:
        print("Adding bin counting features...")
    
    df = df.copy()
    
    # Ensure ts is datetime
    df['ts'] = pd.to_datetime(df['ts'])
    
    # Create skip indicator (1 = skipped/fwdbtn, 0 = not skipped)
    # Handle both original 'reason_end' column and processed 'fwdbtn' column
    if 'fwdbtn' in df.columns:
        df['is_skipped'] = df['fwdbtn'].astype(int)
    elif 'reason_end' in df.columns:
        df['is_skipped'] = df['reason_end'].apply(lambda x: 1 if x == 'fwdbtn' else 0)
    else:
        # For test data without skip information, set all to 0
        df['is_skipped'] = 0
    
    # Define time windows in minutes
    time_windows = {
        '20min': 20,
        '1h': 60, 
        '6h': 360,
        '12h': 720,
        '24h': 1440
    }
    
    # Initialize feature columns
    for window_name in time_windows.keys():
        df[f'songs_total_{window_name}'] = 0
        df[f'songs_skipped_{window_name}'] = 0
    
    # Get unique users and create progress bar
    unique_users = df['username'].unique()
    if verbose:
        print(f"  Processing {len(unique_users)} users...")
        user_progress = unique_users
    else:
        user_progress = unique_users
    
    # Group by user for efficient processing
    for username in tqdm(user_progress, desc="Processing users"):
        # Get user data sorted by timestamp
        user_mask = df['username'] == username
        user_df = df[user_mask].sort_values('ts').copy()
        user_indices = user_df.index.tolist()
        
        # Convert to numpy arrays for faster processing
        timestamps = user_df['ts'].values
        is_skipped = user_df['is_skipped'].values
        
        # For each observation, count songs in time windows using vectorized operations
        for i in tqdm(range(len(user_df))):
            current_time = timestamps[i]
            
            # Calculate time thresholds for all windows
            time_thresholds = [current_time - pd.Timedelta(minutes=window_minutes) 
                              for window_minutes in time_windows.values()]
            
            # Find all songs up to current time (inclusive) - this is the past_songs
            past_mask = timestamps <= current_time
            past_indices = np.where(past_mask)[0]
            
            # For each time window, count songs efficiently
            for j, (window_name, window_minutes) in enumerate(time_windows.items()):
                time_threshold = time_thresholds[j]
                
                # Find songs within the time window (but exclude current song)
                window_mask = (timestamps >= time_threshold) & (timestamps < current_time)
                window_indices = np.where(window_mask)[0]
                
                # Count different types of songs
                total_count = len(window_indices)
                skipped_count = np.sum(is_skipped[window_indices]) if len(window_indices) > 0 else 0
                
                # Assign values back to original dataframe
                original_idx = user_indices[i]
                df.loc[original_idx, f'songs_total_{window_name}'] = total_count
                df.loc[original_idx, f'songs_skipped_{window_name}'] = skipped_count
    
    # Remove temporary columns
    df.drop(columns=['is_skipped'], inplace=True)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        if verbose:
            print(f"Bin counting features cached to: {cache_file}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not save cache file: {e}")
    
    if verbose:
        print(f"  Added {len(time_windows) * 2} bin counting features")
        print(f"  Time windows: {list(time_windows.keys())}")
        print(f"  Feature types: total songs, skipped songs")
    
    return df

def make_features(df: pd.DataFrame, include_username_ohe: bool = True, verbose: bool = False) -> pd.DataFrame:
    df = clean_nulls(df)
    df = add_temporal(df)
    df = add_track_order_in_day(df)
    df = add_track_features(df)
    df = add_platform(df)
    df = add_bin_counting(df, verbose=verbose)
    df.drop(columns=['ts', 'master_metadata_album_artist_name', 'ip_addr','spotify_track_uri', 'spotify_episode_uri', 'Unnamed: 0'], inplace=True)
    return df