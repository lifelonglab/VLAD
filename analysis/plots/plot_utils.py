def filter_by_strategy(df, strategy):
    return df.loc[df['strategy'] == strategy]