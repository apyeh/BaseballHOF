import pandas as pd
import numpy as np
import pickle


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


# Subtract start year from each player's yearID
def subtract_start_yr(row, player_dict):
    name = row['playerID']
    return row['yearID'] - player_dict[name] + 1

# Subtract number of years played up to that point from total number of years played
def get_remaining_yrs(row, player_totalyrs_dict):
    name = row['playerID']
    return player_totalyrs_dict[name] - row['year']

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

def combine_stints(df):
    return df.groupby(['playerID', 'yearID']).sum().reset_index().sort(['playerID', 'yearID'])

def get_cumulative_stats(df):
    # Calculate cumulative stats over the years for each player.
    stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', \
             'IBB', 'HBP', 'SH', 'SF', 'GIDP']
    return df.groupby('playerID')[stats].cumsum()[stats]

def create_yr_col(df):
    # Create 'year' variable indicating the number of years players have played in the MLB.
    player_startyr_dict = pd.DataFrame(df.groupby('playerID').min()['yearID']).to_dict()['yearID']
    df['year'] = df.apply(subtract_start_yr, axis=1, args=(player_startyr_dict,))
    return df

def create_remain_yrs_col(df):
    # Create 'yrs_remain' variable that estimates the remaining number of years for that player.
    # Based on the median number of years that eligible players at that position played
    player_totalyrs_dict = pd.DataFrame(df.groupby('playerID').max()['year']).to_dict()['year']
    df['yrs_remain'] = df.apply(get_remaining_yrs, axis=1, args=(player_totalyrs_dict,))
    return df


# Determine aggregate (i.e., mean, median, or min) of each stat for each position player in HOF

def get_hof_hitter_stats(df, calculate='mean'):
    stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', \
             'IBB', 'HBP', 'SH', 'SF', 'GIDP']

    positions = df['POS'].unique()

    position_stats_lst = []

    if calculate == 'mean':
        stats_labels = [stat + '_mean' for stat in stats]
        stats_labels.append('POS')
        for position in positions:
            pos = pd.Series([position], index=['POS'])
            position_stats = df[(df['inducted'] == 1) & \
                    (df['POS'] == position)].groupby('playerID')[stats].max().mean().round(1).append(pos)
            position_stats_lst.append(position_stats)

    elif calculate == 'median':
        stats_labels = [stat + '_med' for stat in stats]
        stats_labels.append('POS')
        for position in positions:
            pos = pd.Series([position], index=['POS'])
            position_stats = df[(df['inducted'] == 1) & \
                (df['POS'] == position)].groupby('playerID')[stats].max().median().round(1).append(pos)
            position_stats_lst.append(position_stats)

    elif calculate == 'min':
        stats_labels = [stat + '_min' for stat in stats]
        stats_labels.append('POS')
        for position in positions:
            pos = pd.Series([position], index=['POS'])
            position_stats = df[(df['inducted'] == 1) & \
                (df['POS'] == position)].groupby('playerID')[stats].max().min().round(1).append(pos)
            position_stats_lst.append(position_stats)

    hof_hitter_stats_df = pd.DataFrame(position_stats_lst)

    hof_hitter_stats_df.columns = stats_labels
    return hof_hitter_stats_df

# # -------------------------------------------------------------------

if __name__ == '__main__':

    # Select only players from batting table who were/are eligible for HOF. Merge hof labels with batting stats
    elig_labels = get_hof_labels('../data/SeanLahmanBaseballDB/baseballdatabank-master/core/HallOfFame.csv')
    batting = pd.read_csv('../data/SeanLahmanBaseballDB/baseballdatabank-master/core/Batting.csv')
    elig_players = batting.merge(elig_labels, on='playerID')

    # Get and join player positions to the df
    player_pos = get_positions('../data/SeanLahmanBaseballDB/baseballdatabank-master/core/Fielding.csv')
    elig_players = elig_players.merge(player_pos, on='playerID')

    # Select only hitters (remove pitchers from df)
    elig_hitters = elig_players[elig_players['POS'] != 'P']

    # Some players played a stint for different teams in the same season.
    # Combine the stats in those cases into one row.
    elig_hitters = combine_stints(elig_hitters)

    # Create 'year' variable indicating the number of years players have played in the MLB.
    elig_hitters = create_yr_col(elig_hitters)

    # Create 'yrs_remain' variable that estimates the remaining number of years for that player.
    # Based on the median number of years that eligible players at that position played
    elig_hitters = create_remain_yrs_col(elig_hitters)

    # Calculate cumulative stats over the years for each player.
    cumulative_stats = get_cumulative_stats(elig_hitters)

    # Combine cumulative stats with non-stat columns
    cols = list(set(elig_hitters.columns) - set(cumulative_stats.columns))
    #    ['playerID', 'yearID', 'year', 'yrs_remain', 'stint', 'inducted']
    elig_hitters_cumstats = cumulative_stats.join(elig_hitters[cols])

    # Combine cumulative stats with non-stat columns
    cols = list(set(elig_hitters.columns) - set(cumulative_stats.columns))
    #    ['playerID', 'yearID', 'year', 'yrs_remain', 'stint', 'inducted']
    elig_hitters_cumstats = cumulative_stats.join(elig_hitters[cols])


