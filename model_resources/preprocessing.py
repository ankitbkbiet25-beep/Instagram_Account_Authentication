import pandas as pd

def preprocess(df):
    df.drop_duplicates(inplace=True)
    df.rename(columns={
        '#posts': 'posts',
        '#followers': 'followers',
        '#follows': 'follows'
    },inplace=True)
    categorical_cols=['profile pic', 'external URL', 'name==username', 'private']
    df[categorical_cols]=df[categorical_cols].astype(bool)
    df['follower_follow_ratio'] = df['followers'] / (df['follows'] + 1)
    df['follow_post_ratio'] = df['follows'] / (df['posts'] + 1)

    df['SuspicionScore'] = (
        (~df['external URL']).astype(int) +
        (~df['profile pic']).astype(int) +
        (df['name==username']).astype(int) +
        (df['posts'] == 0).astype(int) +
        (df['followers'] < 50).astype(int) +
        ((df['follower_follow_ratio'] < 0.1) & (df['followers'] > 1000)).astype(int) +
        ((df['follow_post_ratio'] > 20) & (df['posts'] < 10)).astype(int) +
        (df['description length'] == 0).astype(int)
    ).astype('int64')

    df['SuspicionLevel'] = pd.cut(
        df['SuspicionScore'],
        bins=[-1, 2, 5, 8],
        labels=['Low', 'Medium', 'High']
    )

    return df
