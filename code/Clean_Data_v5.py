# From Explore_Clean_Data_v10.ipynb (works.. before modifications)

import pandas as pd
import numpy as np
import pickle

def calculate_stat_ratio(row, stat, denom):
    '''
    Description: Calculate the ratio of a player's stats in a particular category to the mean, median, or min of
    that stat for a Hall of Famer at that same position
    '''
    return ((row[stat] / row['year']) * (row['year'] + row['yrs_remain'])) / row[denom]


def create_eras_cols(df):
    # Dead Ball Era 1 (1900-1919)
    df['DBE1'] = 0
    df.ix[(df['yearID'] >= 1900) & (df['yearID'] <= 1919), 'DBE1'] = 1

    # Dead Ball Era 2 (1961-1968)
    df['DBE2'] = 0
    df.ix[(df['yearID'] >= 1961) & (df['yearID'] <= 1968), 'DBE2'] = 1

    # Steroid era (1988-2003)
    df['SE'] = 0
    df.ix[(df['yearID'] >= 1988) & (df['yearID'] <= 2003), 'SE'] = 1
    return df


def combine_stints(df, hof_elig_labels):
    df2 = df.copy()
    if 'inducted' in df2.columns:
        df2 = df2.drop('inducted', axis=1)
    if 'stint' in df2.columns:
        df2 = df2.drop('stint', axis=1)

    df_combine_stints = df2.groupby(['playerID', 'yearID']).sum().reset_index().sort_values(by=['playerID', 'yearID'])
    # Add hof 'inducted' column back to df
    return df_combine_stints.merge(elig_labels, on='playerID')

#def create_stat_ratio_cols(df, stats_of_interest, denominator='mean'):

# def combine_stints(df):
#     return df.groupby(['playerID', 'yearID']).sum().reset_index().sort_values(by=['playerID', 'yearID'])


def create_stat_ratio_cols(df, stats_of_interest, denominator='mean'):
    '''
    Description: Calculate the desired stats ratios and add them as new column to df
    '''
    if denominator == 'mean':
        denom_stats = [stat + '_mean' for stat in stats_of_interest]
    elif denominator == 'median':
        denom_stats = [stat + '_med' for stat in stats_of_interest]
    elif denominator == 'min':
        denom_stats = [stat + '_min' for stat in stats_of_interest]

    stats_ratio = [stat + '_ratio' for stat in stats_of_interest]

    for stat, denom, stat_ratio in zip(stats_of_interest, denom_stats, stats_ratio):
        df[stat_ratio] = df.apply(calculate_stat_ratio, axis=1, args=(stat, denom))
    return df

def create_yr_col(df):
    # Create 'year' variable indicating the number of years players have played in the MLB.
    player_startyr_dict = pd.DataFrame(df.groupby('playerID').min()['yearID']).to_dict()['yearID']
    df['year'] = df.apply(subtract_start_yr, axis=1, args=(player_startyr_dict,))
    return df

def fill_na(df, stats_of_interest):
    stats_filled = []
    for stat in stats_of_interest:
        if df[stat].isnull().sum() > 0:
            stat_filled = stat + '_filled'
            df[stat_filled] = 0
            df.ix[df[stat].isnull(), stat_filled] = 1
            df.ix[df[stat].isnull(), stat] = 0
            stats_filled.append(stat_filled)
    return df, stats_filled

def get_birth_year(filename):
    master = pd.read_csv(filename)
    return pd.DataFrame(master.groupby('playerID').sum()['birthYear']).reset_index()

def get_cumulative_stats(df):
    # Calculate cumulative stats over the years for each player.
    stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', \
             'IBB', 'HBP', 'SH', 'SF', 'GIDP']
    return df.groupby('playerID')[stats].cumsum()[stats]


def get_hof_labels(filename):
    '''
    INPUT: 1 file
    OUTPUT: Pandas df

    Given Hall of Fame (HOF) data file, create HOF labels for all eligible players (inducted and not inducted)
    Returns: dataframe of all eligible HOF players with labels indicating if they were inducted or not.
    '''

    # Load HallofFame.csv file containing players who were/are eligible for election to HOF.
    hof = pd.read_csv(filename)

    # Select those who were inducted into HOF
    hof_players = hof[(hof['inducted'] == 'Y') & (hof['category'] == 'Player')][['playerID', 'inducted']]
    hof_players['inducted'] = hof_players['inducted'].map({'Y' : 1})

#    hof_player_indices = set(hof_players.index)
    hof_playerID = set(hof_players['playerID'])

    # Select all eligible players for the HOF (i.e., those who were on the ballot)
    elig = hof[(hof['category'] == 'Player')]

#    elig_indices = set(elig.index)
    elig_playerID = set(elig['playerID'])

    # Select players who were on the ballot but were not inducted into HOF
    nonhof_playerID = elig_playerID - hof_playerID
    nonhof_playerID = list(nonhof_playerID)
    nonhof_players = pd.DataFrame(nonhof_playerID, columns=['playerID'])
    nonhof_players['inducted'] = 0

    # Merge hof_players and nonhof_players
    return pd.concat([hof_players, nonhof_players])

def get_hofer_stats(df, stats_of_interest, calculate='mean'):
    '''
    Description: Determine mean, median, or min of each stat for HOF players at each position
    '''
#     stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', \
#              'IBB', 'HBP', 'SH', 'SF', 'GIDP']

    positions = df['POS'].unique()
    position_stats_lst = []

    for position in positions:
        pos = pd.Series([position], index=['POS'])
        if calculate == 'mean':
            stats_labels = [stat + '_mean' for stat in stats_of_interest]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats_of_interest].max().mean().round(1).append(pos)
        elif calculate == 'median':
            stats_labels = [stat + '_med' for stat in stats_of_interest]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats_of_interest].max().median().round(1).append(pos)
        elif calculate == 'min':
            stats_labels = [stat + '_min' for stat in stats_of_interest]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats_of_interest].max().min().round(1).append(pos)

        position_stats_lst.append(position_stats)

    hofer_stats_df = pd.DataFrame(position_stats_lst)

    hofer_stats_df.columns = stats_labels
    return hofer_stats_df


def get_positions(filename):

    # Load Fielding.csv file
    fielding = pd.read_csv(filename)

    # Set position of each player to the one at which he played the most games
    fielding_grouped = fielding.groupby(['playerID', 'POS']).sum().reset_index()
    max_game_indices = np.array(fielding_grouped.groupby('playerID')['G'].idxmax())
    player_pos = fielding_grouped.iloc[max_game_indices][['playerID', 'POS']]

    # Set all outfield positions (LF, CF, OF) to OF.
    positions_dict = {'P': 'P', 'OF': 'OF', '1B': '1B', '2B': '2B', 'C': 'C', 'SS': 'SS', \
                    '3B': '3B', 'DH': 'DH', 'CF': 'OF', 'LF': 'OF'}
    player_pos['POS'] = player_pos['POS'].map(positions_dict)

    # # Write out player position dataframe
    # with open('player_pos.pkl', 'w') as f:
    #     pickle.dump(player_pos, f)
    #
    return player_pos


def get_remaining_yrs(row, retire_age='mean'):
    if retire_age == 'mean':
        yrs_remain = row['retire_age_mean'] - row['age']
    elif retire_age == 'median':
        yrs_remain = row['retire_age_med'] - row['age']
    elif retire_age == 'max':
        yrs_remain = row['retire_age_max'] - row['age']
    if yrs_remain >= 0:
        return yrs_remain
    # In case player has played longer than mean career, set to 0.
    else:
        return 0

def get_retirement_age(df, calculate='mean'):
    '''
    Description: Get mean or median retirement age of all eligible players at the different positions
    '''
    positions = df['POS'].unique()
    df_age = pd.DataFrame()

    for position in positions:
        pos = pd.Series([position], index=['POS'])
        if calculate == 'mean':
            retirement_age = round(df[df['POS'] == position].groupby('playerID')['age'].max().mean(), 1)
            col = 'retire_age_mean'
        elif calculate == 'median':
            retirement_age = round(df[df['POS'] == position].groupby('playerID')['age'].max().median(), 1)
            col = 'retire_age_med'
        elif calculate == 'max':
            retirement_age = round(df[df['POS'] == position].groupby('playerID')['age'].max().max(), 1)
            col = 'retire_age_max'

        df_age = df_age.append({'POS': position, col: retirement_age}, ignore_index=True)
    return df_age


def subtract_start_yr(row, player_dict):
    '''
    Description: Subtract start year from each player's yearID
    '''
    name = row['playerID']
    return row['yearID'] - player_dict[name] + 1


# # -------------------------------------------------------------------

if __name__ == '__main__':

    # Select only players from batting table who were/are eligible for HOF.
    elig_labels = get_hof_labels('../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/HallOfFame.csv')

    # Drop a few players from the list (brownwi02, irvinmo01, tennefr01). Willard Brown (brownwi02) and Monte Irvin
    # (irvinmo01) were both inducted into the HOF, but they played the majority of their careers in the Negro League
    # for which stats are not available. Fred Tenney was nominated but only one years worth of stat is available.

    omit = ['brownwi02', 'irvinmo01', 'tennefr01']
    elig_labels = elig_labels[-elig_labels['playerID'].isin(omit)]

    # Also, Jacque Jones' playerID is jonesja05 in the HallofFame file. This corresponds to only one year's worth of stats.
    # Majority of his career stats are associated with playerID jonesja04. Replace with this playerID instead.

    elig_labels.loc[elig_labels['playerID'] == 'jonesja05', 'playerID'] = 'jonesja04'

    # Merge hof labels with batting stats
    batting = pd.read_csv('../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Batting.csv')
    elig_players = batting.merge(elig_labels, on='playerID')

    # Get and join player positions to the df
    player_pos = get_positions('../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Fielding.csv')
    elig_players = elig_players.merge(player_pos, on='playerID')

    # Select only hitters (remove pitchers from df)
    elig_hitters = elig_players[elig_players['POS'] != 'P']

    # Select which stats to include in model
    stats_of_interest = ['R', 'H', 'HR', 'RBI', 'SB', 'BB']

    # Some players played a stint for different teams in the same season.
    # Combine the stats in those cases into one row.
    combined_stints = combine_stints(elig_hitters, elig_labels)

    # Fill in null values that are present in the stats of interest
    filled_na_df, filled_stats = fill_na(combined_stints, stats_of_interest)

    # Calculate cumulative stats over the years for each player.
    cumulative_stats = get_cumulative_stats(filled_na_df)

    # Combine cumulative stats with non-stats columns
    # cols_to_add = list(set(combined_stints.columns) - set(cumulative_stats.columns))
    cols_to_add = list(set(filled_na_df.columns) - set(cumulative_stats.columns))
    elig_hitters_cumstats = cumulative_stats.join(filled_na_df[cols_to_add])[filled_na_df.columns]

    # Get and join birth year of each player to df and create 'age' column
    birth_year = get_birth_year('../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Master.csv')
    elig_hitters_cumstats = elig_hitters_cumstats.merge(birth_year, on='playerID')
    elig_hitters_cumstats['age'] = elig_hitters_cumstats['yearID'] - elig_hitters_cumstats['birthYear']

    # Re-join positions to df
    elig_hitters_cumstats = elig_hitters_cumstats.merge(player_pos, on='playerID')

    # Get mean, median, or min stats of different positions for HOF hitters and merge to df
    hof_hitters_stats = get_hofer_stats(elig_hitters_cumstats, stats_of_interest, 'mean')
    elig_hitters_cumstats = elig_hitters_cumstats.merge(hof_hitters_stats, on='POS')

    # Get mean or median retirement age of different positions for all eligible hitters and merge to df
    retirement_age_elig = get_retirement_age(elig_hitters_cumstats, 'mean')
    elig_hitters_cumstats = elig_hitters_cumstats.merge(retirement_age_elig, on='POS')

    # Get mean or median retirement age of different positions for all MLB players throughout history and merge to df
    # batting_position = batting.merge(player_pos, on='playerID').sort('playerID')
    # batting_position_birthyr = batting_position.merge(birth_year, on='playerID')
    # batting_position_birthyr['age'] = batting_position_birthyr['yearID'] - batting_position_birthyr['birthYear']

    # retirement_age_allMLB = get_retirement_age(batting_position_birthyr, 'max')
    # elig_hitters_cumstats = elig_hitters_cumstats.merge(retirement_age_allMLB, on='POS')

    # Create 'year' variable indicating the number of years players have played in MLB.
    elig_hitters_cumstats = create_yr_col(elig_hitters_cumstats)

    # Create 'yrs_remain' column that estimates the remaining number of years in the career of that player
    # based on the mean retirement age of eligible players at that position
    elig_hitters_cumstats['yrs_remain'] = elig_hitters_cumstats.apply(get_remaining_yrs, axis=1, \
                                                                      args=('mean',))

    # Add variables corresponding to different baseball eras
    elig_hitters_cumstats = create_eras_cols(elig_hitters_cumstats)

    # Calculate the ratio of a player's cumulative total for a particular stat to the mean, median, or min of that
    # stat for players at that position who are in the HOF
    elig_hitters_ratios = create_stat_ratio_cols(elig_hitters_cumstats, stats_of_interest, 'mean')



    # Select the stat ratio columns as feature set on which to train model
    features = [stat + '_ratio' for stat in stats_of_interest] + filled_stats
    X = elig_hitters_ratios[features]

    features2 = features = [stat + '_ratio' for stat in stats_of_interest] + ['year'] + filled_stats
    X2 = elig_hitters_ratios[features2]

    features3 = features = [stat + '_ratio' for stat in stats_of_interest] + ['year', 'yearID'] + filled_stats
    X3 = elig_hitters_ratios[features3]

    features4 = features = [stat + '_ratio' for stat in stats_of_interest] + ['year', 'yearID', 'DBE1',
                                                                             'DBE2', 'SE'] + filled_stats
    X4 = elig_hitters_ratios[features4]


    # Select 'inducted' column as target variable (1 = inducted into HOF, 0 = not inducted into HOF)
    y = elig_hitters_ratios['inducted']

    # Write out feature and label data
    with open('X_features_hitters.pkl', 'w') as f:
        pickle.dump(X, f)

    with open('X2_features_hitters.pkl', 'w') as f:
        pickle.dump(X2, f)

    with open('X3_features_hitters.pkl', 'w') as f:
        pickle.dump(X3, f)

    with open('X4_features_hitters.pkl', 'w') as f:
        pickle.dump(X4, f)

    with open('y_labels_hitter.pkl', 'w') as f:
        pickle.dump(y, f)
