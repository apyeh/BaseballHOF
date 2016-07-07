import pandas as pd
import numpy as np
import pickle

# ==========================================================================================

# Functions

def add_allstars(df, allstar_file):
    '''
    Description: add all-star game column
    '''
    # Read in allstars file
    allstars = pd.read_csv(allstar_file)
    # Remove duplicate years
    allstars = allstars.groupby(['playerID', 'yearID']).sum().reset_index()
    # Create allstar column
    allstars['AS'] = 1
    # Merge allstars with input df
    df_merged = pd.merge(df, allstars[['playerID', 'yearID', 'AS']],\
                  how='left', on=['playerID', 'yearID'])
    # Fill NaNs with 0
    df_merged['AS'].fillna(0, inplace=True)
    return df_merged


def add_awards(df, awards_file):
    '''
    Description: Add MVP, Triple Crown, and Gold Glove awards.
    '''
    # Read in awards csv file
    awards = pd.read_csv(awards_file)

    # Select only MVP, Triple Crown, and Gold Glove awards
    awards_subset = awards[awards['awardID'].isin(['Most Valuable Player', 'Triple Crown', 'Gold Glove'])]
    awards_subset = pd.concat([awards_subset, pd.get_dummies(awards_subset['awardID'])], axis=1)
    awards_subset.rename(columns={'Most Valuable Player': 'MVP'}, inplace=True)
    awards_subset = awards_subset.groupby(['playerID', 'yearID']).sum().reset_index()

    # Merge awards_subset with df from argument
    df_merged = pd.merge(df, awards_subset[['playerID', 'yearID', 'Gold Glove', 'MVP', 'Triple Crown']],\
                  how='left', on=['playerID', 'yearID'])

    # Fill NaNs with 0
    df_merged['Gold Glove'].fillna(0, inplace=True)
    df_merged['MVP'].fillna(0, inplace=True)
    df_merged['Triple Crown'].fillna(0, inplace=True)
    return df_merged


def add_eras_cols(df):
    '''
    Description: add the different baseball eras to the feature matrix

    '''
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


def add_stat_ratio_cols(df, stats, denominator='mean'):
    '''
    Description: Calculate the desired stats ratios and add them as new column to df
    '''
    if denominator == 'mean':
        denom_stats = [stat + '_mean' for stat in stats]
    elif denominator == 'median':
        denom_stats = [stat + '_med' for stat in stats]
    elif denominator == 'min':
        denom_stats = [stat + '_min' for stat in stats]

    stats_ratio = [stat + '_ratio' for stat in stats]

    df2 = df.copy()
    for stat, denom, stat_ratio in zip(stats, denom_stats, stats_ratio):
        df2[stat_ratio] = df2.apply(calculate_stat_ratio, axis=1, args=(stat, denom))
    return df2


def add_yr_col(df):
    '''
    Description: Add a year column corresponding to the number of years players have played in the MLB.
    '''
    player_startyr_dict = pd.DataFrame(df.groupby('playerID').min()['yearID']).to_dict()['yearID']
    df['year'] = df.apply(subtract_start_yr, axis=1, args=(player_startyr_dict,))
    return df


def calculate_stat_ratio(row, stat, denom):
    '''
    Description: Calculate the ratio of a player's stats in a particular category to the mean,
    median, or min of that stat for a Hall of Famer at that same position
    '''
    return ((row[stat] / row['year']) * (row['year'] + row['yrs_remain'])) / row[denom]


def combine_stints(df):
    '''
    Description: Some players played for multiple teams in the same season (multiple stints).
    Combine the stats from the different stints into one row.
    '''

    df2 = df.copy()
    if 'inducted' in df2.columns:
        df2.drop('inducted', axis=1, inplace=True)
    if 'stint' in df2.columns:
        df2.drop('stint', axis=1, inplace=True)
    return df2.groupby(['playerID', 'yearID']).sum().reset_index().sort_values(by=['playerID', 'yearID'])


def fill_na(df, stats):
    '''
    Description: fill stats with null values with zero.

    '''
    stats_filled = []
    for stat in stats:
        if df[stat].isnull().sum() > 0:
            stat_filled = stat + '_filled'
            df[stat_filled] = 0
            df.ix[df[stat].isnull(), stat_filled] = 1
            df.ix[df[stat].isnull(), stat] = 0
            stats_filled.append(stat_filled)
    return df, stats_filled


def get_avg_position_player_stats(df, features):
    '''
    Description: Obtain the average stats for a position player
    '''

    positions = df['POS'].unique()
    position_stats_lst = []

    for position in positions:
        pos = pd.Series([position], index=['POS'])
        position_stats = df[df['POS'] == position][features].mean().append(pos)
        position_stats_lst.append(position_stats)
    nonhofer_stats_df = pd.DataFrame(position_stats_lst)
    return nonhofer_stats_df.set_index('POS')


def get_birth_year(filename):
    '''
    Description: return a df of the birth years of players
    '''
    master = pd.read_csv(filename)
    return pd.DataFrame(master.groupby('playerID').sum()['birthYear']).reset_index()

def get_name(filename):
    '''
    Description: merge the names of the players to the df
    '''
    master = pd.read_csv(filename)
    df = master[['playerID', 'nameFirst', 'nameLast']]
    df['name'] = df['nameFirst'].map(str) + ' ' + df['nameLast'].map(str)
    df.drop(['nameFirst', 'nameLast'], axis=1, inplace=True)
    return df

def get_cumulative_stats(df, stats_to_accumulate):
    '''
    Description: Calculate cumulative stats over the years for each player.
    '''
    return df.groupby('playerID')[stats_to_accumulate].cumsum()[stats_to_accumulate]


def get_hof_labels(filename):
    '''
    INPUT: HallofFame.csv
    OUTPUT: Pandas df

    Description: Given Hall of Fame (HOF) data file, create HOF labels for all eligible players
    (both inducted and not inducted)

    Returns: dataframe of all eligible HOF players with labels indicating if they were inducted or not.
    '''

    # Load HallofFame.csv file containing players who were/are eligible for election to HOF.
    hof = pd.read_csv(filename)

    # Select those who were inducted into HOF
    hof_players = hof[(hof['inducted'] == 'Y') & (hof['category'] == 'Player')][['playerID', 'inducted']]
    hof_players['inducted'] = hof_players['inducted'].map({'Y' : 1})

    hof_playerID = set(hof_players['playerID'])

    # Select all eligible players for the HOF (i.e., those who were on the ballot)
    elig = hof[(hof['category'] == 'Player')]

    elig_playerID = set(elig['playerID'])

    # Select players who were on the ballot but were not inducted into HOF
    nonhof_playerID = elig_playerID - hof_playerID
    nonhof_playerID = list(nonhof_playerID)
    nonhof_players = pd.DataFrame(nonhof_playerID, columns=['playerID'])
    nonhof_players['inducted'] = 0

    # Merge hof_players and nonhof_players
    return pd.concat([hof_players, nonhof_players])


def get_hofer_stats(df, stats, calculate='mean'):
    '''
    Description: Determine mean, median, or min of each stat for HOF players at each position
    '''

    positions = df['POS'].unique()
    position_stats_lst = []

    for position in positions:
        pos = pd.Series([position], index=['POS'])
        if calculate == 'mean':
            stats_labels = [stat + '_mean' for stat in stats]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats].max().mean().round(1).append(pos)
        elif calculate == 'median':
            stats_labels = [stat + '_med' for stat in stats]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats].max().median().round(1).append(pos)
        elif calculate == 'min':
            stats_labels = [stat + '_min' for stat in stats]
            stats_labels.append('POS')
            position_stats = df[(df['inducted'] == 1) & (df['POS'] == position)]\
            .groupby('playerID')[stats].max().min().round(1).append(pos)

        position_stats_lst.append(position_stats)

    hofer_stats_df = pd.DataFrame(position_stats_lst)

    hofer_stats_df.columns = stats_labels
    return hofer_stats_df


def get_positions(filename):
    '''
    Description: Get the position at which each player played the majority of his games
    '''

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

    return player_pos


def get_remaining_yrs(row, retire_age='mean'):
    '''
    Description: calculate and return the projected remaining years of a player's career.
    '''

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

if __name == '__main__':
    main()


def main():

    allstar_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/AllstarFull.csv'
    awards_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/AwardsPlayers.csv'
    batting_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Batting.csv'
    fielding_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Fielding.csv'
    hall_of_fame_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/HallOfFame.csv'
    master_file = '../BaseballHOF-repo/data/SeanLahmanBaseballDB/baseballdatabank-master/core/Master.csv'

    # Get and join player positions to the df
    player_pos = get_positions(fielding_file)
    batting = pd.read_csv(batting_file)
    all_players = batting.merge(player_pos, on='playerID')

    # Select only hitters (remove pitchers from df)
    all_hitters = all_players[all_players['POS'] != 'P']

    # Some players played a stint for different teams in the same season.
    # Combine the stats in those cases into one row.
    combined_stints = combine_stints(all_hitters)

    # Add awards
    awards_df = add_awards(combined_stints, awards_file)

    # Select which stats to create ratios for
    stats_to_ratio = ['R', 'H', 'HR', 'RBI', 'SB', 'BB']

    # Fill in null values that are present in the stats of interest
    filled_na_df, filled_stats = fill_na(awards_df, stats_to_ratio)

    # Calculate cumulative stats over the years for each player.
    stats_to_accumulate = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', \
                 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'MVP', 'Triple Crown', 'Gold Glove']

    cumulative_stats = get_cumulative_stats(filled_na_df, stats_to_accumulate)

    # Combine cumulative stats with non-stats columns
    cols_to_add = list(set(filled_na_df.columns) - set(cumulative_stats.columns))
    all_hitters_cumstats = cumulative_stats.join(filled_na_df[cols_to_add])[filled_na_df.columns]

    # Get and join birth year, first and last name of each player to df and create 'age' column
    birth_year = get_birth_year(master_file)
    all_hitters_cumstats = all_hitters_cumstats.merge(birth_year, on='playerID')
    all_hitters_cumstats['age'] = all_hitters_cumstats['yearID'] - all_hitters_cumstats['birthYear']

    # Create 'year' variable indicating the number of years players have played in MLB.
    all_hitters_cumstats = add_yr_col(all_hitters_cumstats)

    # Positions column is lost when the stats are accumulated above. Re-join positions to df
    all_hitters_cumstats = all_hitters_cumstats.merge(player_pos, on='playerID')

    # Add names of players.
    names = get_name(master_file)
    all_hitters_cumstats = all_hitters_cumstats.merge(names, on='playerID')

    # Add variables corresponding to different baseball eras
    all_hitters_cumstats = add_eras_cols(all_hitters_cumstats)

    # --------------------------------------------------------------------------------------------

    # Create dataframe of HOF eligible hitters.

    # Select only players from batting table who were/are eligible for HOF.
    elig_labels = get_hof_labels(hall_of_fame_file)


    # Drop a few players from the list (brownwi02, irvinmo01, tennefr01, rosepe01). Willard Brown (brownwi02)
    # and Monte Irvin (irvinmo01) were both inducted into the HOF, but they played the majority of their careers
    # in the Negro League for which stats are not available. Fred Tenney was nominated but only one years worth
    # of stat is available. Pete Rose, in all likelihood, would be in HOF if not for scandal.

    omit = ['brownwi02', 'irvinmo01', 'tennefr01', 'rosepe01']
    elig_labels = elig_labels[-elig_labels['playerID'].isin(omit)]

    # Also, Jacque Jones' playerID is jonesja05 in the HallofFame file. This corresponds to only one year's worth of stats.
    # Majority of his career stats are associated with playerID jonesja04. Replace with this playerID instead.

    elig_labels.loc[elig_labels['playerID'] == 'jonesja05', 'playerID'] = 'jonesja04'


    # Merge hof labels with all_hitters_cumstats df
    elig_hitters_cumstats = all_hitters_cumstats.merge(elig_labels, on='playerID')

    # Get mean, median, or min stats of different positions for HOF hitters and merge to dfs
    hof_hitters_stats = get_hofer_stats(elig_hitters_cumstats, stats_to_ratio, 'mean')

    # Merge avg hof hitters stats to dataframe containing the cumulative stats
    all_hitters_cumstats = all_hitters_cumstats.merge(hof_hitters_stats, on='POS')
    elig_hitters_cumstats = elig_hitters_cumstats.merge(hof_hitters_stats, on='POS')

    # Get mean or median retirement age of different positions for all eligible HOF hitters and merge to dfs
    retirement_age_elig = get_retirement_age(elig_hitters_cumstats, 'mean')

    all_hitters_cumstats = all_hitters_cumstats.merge(retirement_age_elig, on='POS')
    elig_hitters_cumstats = elig_hitters_cumstats.merge(retirement_age_elig, on='POS')


    # --------------------------------------------------------------------------------------------

    # Create 'yrs_remain' column that estimates the remaining number of years in the career of that player
    # based on the mean retirement age of eligible players at that position
    all_hitters_cumstats['yrs_remain'] = all_hitters_cumstats.apply(get_remaining_yrs, axis=1, \
                                                                      args=('mean',))
    elig_hitters_cumstats['yrs_remain'] = elig_hitters_cumstats.apply(get_remaining_yrs, axis=1, \
                                                                      args=('mean',))

    # Calculate the ratio of a player's cumulative total for a particular stat to the mean, median,
    # or min of that stat for players at that position who are in the HOF
    all_hitters_ratios = add_stat_ratio_cols(all_hitters_cumstats, stats_to_ratio, 'mean')
    elig_hitters_ratios = add_stat_ratio_cols(elig_hitters_cumstats, stats_to_ratio, 'mean')


    # --------------------------------------------------------------------------------------------

    # Write out files.

    # Filter all_hitters_ratios df for just active players and those still on the HOF ballot

    # Get most current year of database
    most_current_yr = all_hitters_ratios[['playerID', 'yearID']].groupby('playerID').max()['yearID'].max()

    # Get the last year in which a player has played
    all_hitters_ratios_last_yr = all_hitters_ratios[['playerID', 'yearID']].groupby('playerID').max().reset_index()

    # Select only for hitters that are currently still in MLB
    active_hitters = all_hitters_ratios_last_yr[all_hitters_ratios_last_yr['yearID'] == most_current_yr]['playerID'].values
    active_hitters_ratios = all_hitters_ratios[all_hitters_ratios['playerID'].isin(active_hitters)]

    with open('active_hitters.pkl', 'w') as f:
        pickle.dump(active_hitters_ratios[['playerID', 'name', 'yearID', 'year']], f)

    # Select for hitters that have recently been in MLB (in the past 10 years)
    recent_hitters = all_hitters_ratios_last_yr[(all_hitters_ratios_last_yr['yearID'] >= (most_current_yr - 10)) & \
                                               (all_hitters_ratios_last_yr['yearID'] < most_current_yr)]['playerID'].values
    recent_hitters_ratios = all_hitters_ratios[all_hitters_ratios['playerID'].isin(recent_hitters)]

    with open('recent_hitters.pkl', 'w') as f:
        pickle.dump(recent_hitters_ratios[['playerID', 'name', 'yearID']], f)

    # --------------------------------------------------------------------------------------------

    # Select the stat ratio columns as feature set on which to train model and write out.

    eras = ['DBE1', 'DBE2', 'SE']
    awards = ['MVP', 'Triple Crown', 'Gold Glove']

    # Feature set
    features = [stat + '_ratio' for stat in stats_to_ratio] + eras + filled_stats + awards
    X = elig_hitters_ratios[features]
    with open('eligible_hitters_X.pkl', 'w') as f:
        pickle.dump(X, f)

    # Select 'inducted' column as target variable (1 = inducted into HOF, 0 = not inducted into HOF)
    y = elig_hitters_ratios['inducted']
    with open('eligible_hitters_y.pkl', 'w') as f:
        pickle.dump(y, f)

    # Select the stat ratio columns as feature set for the active and recent players
    active = active_hitters_ratios[features]
    with open('active_hitters_X.pkl', 'w') as f:
        pickle.dump(active, f)

    recent = recent_hitters_ratios[features]
    with open('recent_hitters_X.pkl', 'w') as f:
        pickle.dump(recent, f)
