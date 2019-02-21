import numpy
import pandas
import warnings
import datetime
import matplotlib.pyplot as plt
import basketball_reference_scraper
# from sklearn.datasets import load_iris
# from sklearn.neighbors import KNeighborsClassifier
from progressbar import ProgressBar, Bar, Percentage, ETA, FileTransferSpeed

warnings.filterwarnings("ignore")

# string translation
win_loss_str  = ['W', 'L']
win_loss_val  = [ 0 ,  1 ]
home_away_str = ['Home', 'Away']
home_away_val = [   0  ,    1  ]
teams_str     = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
teams_val     = [  0  ,   1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ,   8  ,   9  ,   10 ,   11 ,   12 ,   13 ,   14 ,   15 ,   16 ,   17 ,   18 ,   19 ,   20 ,   21 ,   22 ,   23 ,   24 ,   25 ,   26 ,   27 ,   28 ,   29 ]

# import datasets
season_years = '2018-2019'
nba = pandas.read_csv('Data/NBA_'+ season_years + '_Data.csv')
weights = pandas.read_csv('Weights/weights_'+ season_years + '.csv')

# categories
original_categories = ['TEAM','DATE','MATCHUP','W/L','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','PF','+/-']
input_categories    = ['PTS Avg.', 'FGM Avg.', 'FGA Avg.', 'FG% Avg.', '3PM Avg.', '3PA Avg.', '3P% Avg.', 'FTM Avg.', 'FTA Avg.', 'FT% Avg.', 'OREB Avg.', 'DREB Avg.', 'REB Avg.', 'AST Avg.', 'STL Avg.', 'BLK Avg.', 'TOV Avg.', 'PF Avg.', '+/- Avg.', 'Opp. PTS Avg.', 'Opp. FGM Avg.', 'Opp. FGA Avg.', 'Opp. FG% Avg.', 'Opp. 3PM Avg.', 'Opp. 3PA Avg.', 'Opp. 3P% Avg.', 'Opp. FTM Avg.', 'Opp. FTA Avg.', 'Opp. FT% Avg.', 'Opp. OREB Avg.', 'Opp. DREB Avg.', 'Opp. REB Avg.', 'Opp. AST Avg.', 'Opp. STL Avg.', 'Opp. BLK Avg.', 'Opp. TOV Avg.', 'Opp. PF Avg.', 'Opp. +/- Avg.']

# constants
THOUSAND = 10**3
MILLION = 10**6

def main():
    # # ------------------------------ Testing - One Game ------------------------------- #
    # widgets = [Bar(marker='=',left='[',right=']'), ' ', Percentage(), ' ', ETA(), ' ', FileTransferSpeed()]
    # progress_bar = ProgressBar(widgets=widgets, maxval=len(teams_str))
    # progress_bar.start()
    # correct = 0
    # wrong_teams = []
    # for team_num in range(len(teams_str)):
    #     progress_bar.update(team_num)
        
    #     # team data
    #     team = getTeamGames(nba, teams_str[team_num])
    #     team = addDefensiveStats(team)
    #     team = addStatAverages(team)

    #     testing_inputs = numpy.array([team[input_categories].iloc[81].values])
    #     weight = weights[weights['team'] == teams_str[team_num]][input_categories].T
    #     output = sigmoid(numpy.dot(testing_inputs, weight))[0]
    #     actual_output = team['W/L'].iloc[81]
    #     if (output == 1 and actual_output == 'W') or (output == 0 and actual_output == 'L'):
    #         correct = correct + 1
    #     else:
    #         wrong_teams.append(teams_str[team_num])
    # progress_bar.finish()

    # # --------------------------- Testing Output - One Game  -------------------------- #
    # print (str(float(correct)/len(teams_str))[:5])
    # print (wrong_teams)

    # # ------------------------------ Testing - Many Games ------------------------------- #
    # widgets = [Bar(marker='=',left='[',right=']'), ' ', Percentage(), ' ', ETA(), ' ', FileTransferSpeed()]
    # progress_bar = ProgressBar(widgets=widgets, maxval=len(teams_str))
    # progress_bar.start()
    # correct = 0
    # wrong_teams = []
    # for team_num in range(len(teams_str)):
    #     progress_bar.update(team_num)
        
    #     # team data
    #     team = getTeamGames(nba, teams_str[team_num])
    #     team = addDefensiveStats(team)
    #     team = addStatAverages(team)

    #     testing_inputs = numpy.array([team[input_categories].iloc[70:81].values])
    #     weight = weights[weights['team'] == teams_str[team_num]][input_categories].T
    #     output = sigmoid(numpy.dot(testing_inputs, weight))
    #     actual_output = team['W/L'].iloc[70:81]
    #     if (output == 1 and actual_output == 'W') or (output == 0 and actual_output == 'L'):
    #         correct = correct + 1
    #     else:
    #         wrong_teams.append(teams_str[team_num])
    # progress_bar.finish()

    # # --------------------------- Testing Output - Many Games  -------------------------- #
    # print (str(float(correct)/len(teams_str))[:5])
    # print (wrong_teams)

    
    
    # ---------------------------------- Training  ------------------------------------ #
    # team data
    new_weights = {}
    for team_str in teams_str:
        team = getTeamGames(nba, team_str)
        team = addDefensiveStats(team)
        team = addStatAverages(team)
        print('\n', team_str)

        training_inputs = numpy.array(team[input_categories].iloc[10:70].values)
        training_outputs = numpy.array([team['W/L'].iloc[10:70].values])

        # translate wins and losses into 1's and 0's 
        training_outputs[training_outputs == 'W'] = 1
        training_outputs[training_outputs == 'L'] = 0
        training_outputs = numpy.array(training_outputs).T

        numpy.random.seed(1)
        random_weight = numpy.random.random((len(training_inputs[0]), 1))
        syn_weights = 2 * random_weight - 1

        widgets = [Bar(marker='=',left='[',right=']'), ' ', Percentage(), ' ', ETA(), ' ', FileTransferSpeed()]
        range_val = MILLION

        team_weights = numpy.array([weights[weights['team'] == team_str].iloc[0].values[1:]]).T
        new_team_weights = trainTeam(training_inputs, training_outputs, team_weights, acceptable_accuracy=85)
        new_weights[team_str] = new_team_weights

    new_weights_str = 'team,PTS Avg.,FGM Avg.,FGA Avg.,FG% Avg.,3PM Avg.,3PA Avg.,3P% Avg.,FTM Avg.,FTA Avg.,FT% Avg.,OREB Avg.,DREB Avg.,REB Avg.,AST Avg.,STL Avg.,BLK Avg.,TOV Avg.,PF Avg.,+/- Avg.,Opp. PTS Avg.,Opp. FGM Avg.,Opp. FGA Avg.,Opp. FG% Avg.,Opp. 3PM Avg.,Opp. 3PA Avg.,Opp. 3P% Avg.,Opp. FTM Avg.,Opp. FTA Avg.,Opp. FT% Avg.,Opp. OREB Avg.,Opp. DREB Avg.,Opp. REB Avg.,Opp. AST Avg.,Opp. STL Avg.,Opp. BLK Avg.,Opp. TOV Avg.,Opp. PF Avg.,Opp. +/- Avg.\n'
    for team_str in teams_str:
        new_weights_str = new_weights_str + (team_str + ',' + ','.join(map(str, numpy.array(new_weights[team_str]).T[0]))) + '\n'

    # -------------------------------- Training Output -------------------------------- #
    # print('\n', new_weights, '\n\n')

    # writes to csv
    with open('Weights/weights_' + season_years + '.csv', 'w') as filetowrite:
        filetowrite.write(new_weights_str)
        filetowrite.close()

# training function
def trainTeam(training_inputs, training_outputs, weights, acceptable_accuracy=85):
    accuracy = 0
    iterations = 0
    syn_weights = weights
    while(accuracy < acceptable_accuracy):
        input_layer = training_inputs
        outputs = sigmoid(numpy.dot(input_layer, syn_weights))
        error = training_outputs - outputs
        adjustments = error * sigmoid_prime(outputs)
        syn_weights = syn_weights + numpy.dot(input_layer.T, adjustments)
        accuracy = float(getAccuracyPercentage(outputs, training_outputs))
        iterations = iterations + 1
    print('Accuracy: ' + getAccuracyPercentage(outputs, training_outputs) + '%\n')
    return syn_weights

# Activation Functions
def sigmoid(x):
    return 1 / (1 + numpy.exp(-(x.astype(float))))
def sigmoid_prime(x):
    return sigmoid(x.astype(float)) * (1 - sigmoid(x.astype(float)))
def sigmoid_derivative(x):
    return x.astype(float) * (1 - x.astype(float))

# populate data
def addDefensiveStats(data):
    # make copy of data
    new_data = pandas.DataFrame()
    for item in data.keys().tolist():
            new_data[item] = data[item]

    # new categories
    opp_teams  = []
    opp_stats = {}

    # get categories
    categories = []
    ignore_categories = ['TEAM', 'DATE', 'MATCHUP', 'W/L', 'MIN']
    for x in data:
        if x[-4:] != 'Avg.' and x not in ignore_categories:
            categories.append(x)

    # fill opp_stats
    for x in categories:
        opp_stats[x] = []
    
    # get opponents stat dictionary
    for i in new_data['TEAM'].keys():
        team_name = new_data['TEAM'].loc[i]
        opponent_name = teams_str[translateMatchup(new_data['MATCHUP'].loc[i])[1]]
        date = new_data['DATE'].loc[i]
        getOpponentData(team_name, opponent_name, date)
        opp_df = getOpponentData(team_name, opponent_name, date)
        for x in opp_stats:
            opp_stats[x].append(opp_df[x].iloc[0])
    
    # add opponent stats to new_data
    for x in opp_stats:
        new_data['Opp. ' + x] = opp_stats[x]
    return(new_data)
def addStatAverages(data):
    data_first10_avg = getAverages(data[:10])
    
    # make copy of data
    new_data = pandas.DataFrame()
    for item in data.keys().tolist():
            new_data[item] = data[item]

    for index in data:
        i = 0
        avgs = []
        if index in data_first10_avg.keys():
            for i in range(0, 10):
                avgs.append(0)
            for i in range(0, len(new_data[index])-10):
                i = i + 1
                avgs.append(data[index].iloc[:10+i].mean())
            new_data[index + ' Avg.'] = avgs
    
    # return copy
    return(new_data)   

# retrieval
def getWins(data, criteria=None):
    if criteria is None:
        return(data[data['W/L'].str.contains('W')])
    else:
        return(data[data['W/L'].str.contains('W')])[criteria]
def getLosses(data, criteria=None):
    if criteria is None:
        return(data[data['W/L'].str.contains('L')])
    else:
        return(data[data['W/L'].str.contains('L')])[criteria]
def getHome(data, criteria=None):
    if criteria is None:
        return data[data['MATCHUP'].str.contains('vs.')]
    else:
        return data[data['MATCHUP'].str.contains('vs.')][criteria]
def getAway(data, criteria=None):
    if criteria is None:
        return data[data['MATCHUP'].str.contains('@')]
    else:
        return data[data['MATCHUP'].str.contains('@')][criteria]
def getTeamGames(data, team_name, criteria=None):
    if criteria is None:
        return(data[data['TEAM'].str.contains(team_name)])
    else:
        return(data[data['TEAM'].str.contains(team_name)])[criteria]
def getAverages(data):
    results = {}
    ignore_categories = ['TEAM', 'DATE', 'MATCHUP', 'W/L', 'MIN']
    for stat in data.keys().tolist():
        if stat not in ignore_categories and stat[-4:] != 'Avg.':
            results[stat] = data[stat].mean()
    return results
def getOpponentData(team, opponent, date, data=nba):
    return(data[data['MATCHUP'].str.contains(team) & data['MATCHUP'].str.contains(opponent) & data['DATE'].str.contains(date) & data['TEAM'].str.contains(opponent)])

#translation
def translateMatchup(matchup, win_loss=None):
    team = teams_str.index(matchup[:3])
    opp  = teams_str.index(matchup[-3:])
    if (matchup[4] == '@'): home_away = 1
    else: home_away = 0
    if win_loss:
        result = win_loss_str.index(win_loss)
        return [team, opp, result, home_away]
    else:
        return [team, opp, home_away]
def translatePredictions(array):
    new_array = []
    for i in array: 
        if i == 'W':
            new_array.append(1)
        elif i == 'L':
            new_array.append(0)
        else:
            print('error with translating predictions')
    return new_array

# calculate accuracy 
def getAccuracyPercentage(calculated_results, actual_results):
    total = len(actual_results.T[0])
    right = 0
    for i in range(total):
        if actual_results.T[0][i] == calculated_results[i][0]:
            right = right + 1
    return str(right/total*100)[:5]

if __name__ == '__main__':
    main()
