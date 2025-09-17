import pandas as pd
import numpy as np

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

def make_features(df: pd.DataFrame, include_username_ohe: bool = True, verbose: bool = False) -> pd.DataFrame:
    df = clean_nulls(df)
    df = add_temporal(df)
    df = add_track_order_in_day(df)
    df = add_track_features(df)
    df = add_platform(df)
    
    df.drop(columns=['ts', 'master_metadata_album_artist_name', 'ip_addr','spotify_track_uri', 'spotify_episode_uri', 'Unnamed: 0'], inplace=True)
    return df