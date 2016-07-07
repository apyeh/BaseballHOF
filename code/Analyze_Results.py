# df1 = elig_hitters_ratios
# df2 = all_hitters_ratios

def find_similar_players(name, age, position, df1, df2, features, num=10):
    '''
    Description: Using cosine similarity, find HOF players that are most similar to player in question at a
    particular age.

    INPUT:
    name, age, and position of player of interest
    df1: the dataframe containing the stats of all HOF eligible players
    df2: the dataframe containing the stats of all hitters
    features: the stats by which to compare players
    num: the number of most similar HOFers to return

    OUTPUT: dataframe that lists the most similar HOF players using cosine similarity
    '''
    # Get the stats of all HOF players at the same position and same age as player of interest
    hofers = df1[(df1['inducted'] == 1) & (df1['age'] == age) & (df1['POS'] == position)][features]

    # Get the stats of player of interest
    player_stats = df2[(df2['name'] == name) & (df2['age'] == age)][features]

    # Calculate cosine similarities between player of interest and HOF players
    cossim = cosine_similarity(np.array(player_stats), np.array(hofers))
    cossim_df = pd.DataFrame(cossim[0], index=hofers.index, columns=['cossim'])

    #
    cossim_merged = df1.merge(cossim_df, left_index=True, right_index=True)
    return cossim_merged[['name', 'yearID', 'cossim']].sort_values(by='cossim', ascending=False).head(num)


# df = nonhof_hitters_ratios_max

def get_avg_position_player_stats(df, features):
    '''
    Description: Obtain the average stats for a position player
    '''

    positions = df['POS'].unique()
    position_stats_lst = []

#     df_stats = pd.DataFrame(df[df['POS'] == '3B'][features].mean()).transpose()
#     df_stats.merge(pos, left_index=True, right_index=True)
#     df[df['POS'] == '3B'].groupby('playerID')[features].mean()

    for position in positions:
#        pos = pd.DataFrame([position], index=['POS']).transpose()
        pos = pd.Series([position], index=['POS'])
        position_stats = df[df['POS'] == position][features].mean().append(pos)
        position_stats_lst.append(position_stats)
#        print position_stats_lst
    nonhofer_stats_df = pd.DataFrame(position_stats_lst)
#    print nonhofer_stats_df
#    nonhofer_stats_df.columns = features
    return nonhofer_stats_df.set_index('POS')

    #        print 'pos: ', pos
#        df_stats = pd.DataFrame(df[df['POS'] == position][features].mean()).transpose()

        #        print 'df_stats: ', df_stats
    #    position_stats = df_stats.merge(pos, left_index=True, right_index=True)


def get_nonhofer_stats(df, stats_of_interest, calculate='mean'):
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
            position_stats = df[df['POS'] == position]\
            .groupby('playerID')[stats_of_interest].max().mean().round(1).append(pos)
            print position_stats
        elif calculate == 'median':
            stats_labels = [stat + '_mean' for stat in stats_of_interest]
            stats_labels.append('POS')
            position_stats = df[df['POS'] == position]\
            .groupby('playerID')[stats_of_interest].max().median().round(1).append(pos)
            print position_stats
        elif calculate == 'min':
            stats_labels = [stat + '_mean' for stat in stats_of_interest]
            stats_labels.append('POS')
            position_stats = df[df['POS'] == position]\
            .groupby('playerID')[stats_of_interest].max().min().round(1).append(pos)

        position_stats_lst.append(pos)

    nonhofer_stats_df = pd.DataFrame(position_stats_lst)

#    nonhofer_stats_df.columns = stats_labels
    return nonhofer_stats_df, stats_labels, position_stats_lst


if __name__ == '__main__':


    # Determine the probability of the average position player to be inducted into HOF
    idx_max = nonhof_hitters_ratios.groupby('name')['year'].transform(max) == nonhof_hitters_ratios['year']
    nonhof_hitters_ratios_max = nonhof_hitters_ratios[idx_max].groupby('name').max()
    avg9 = get_avg_position_player_stats(nonhof_hitters_ratios_max, features9)


    average9 = get_avg_position_player_stats(nonhof_hitters_ratios_max, features9)
