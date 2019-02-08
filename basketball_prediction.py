import numpy
import pandas
import warnings
import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
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
nba2017 = pandas.read_csv('Data/NBA_2017-2018_Data.csv')
weights = pandas.read_csv('Data/weights.csv')

# categories
original_categories = ['TEAM','DATE','MATCHUP','W/L','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','PF','+/-']
input_categories    = ['PTS Avg.', 'FGM Avg.', 'FGA Avg.', 'FG% Avg.', '3PM Avg.', '3PA Avg.', '3P% Avg.', 'FTM Avg.', 'FTA Avg.', 'FT% Avg.', 'OREB Avg.', 'DREB Avg.', 'REB Avg.', 'AST Avg.', 'STL Avg.', 'BLK Avg.', 'TOV Avg.', 'PF Avg.', '+/- Avg.', 'Opp. PTS Avg.', 'Opp. FGM Avg.', 'Opp. FGA Avg.', 'Opp. FG% Avg.', 'Opp. 3PM Avg.', 'Opp. 3PA Avg.', 'Opp. 3P% Avg.', 'Opp. FTM Avg.', 'Opp. FTA Avg.', 'Opp. FT% Avg.', 'Opp. OREB Avg.', 'Opp. DREB Avg.', 'Opp. REB Avg.', 'Opp. AST Avg.', 'Opp. STL Avg.', 'Opp. BLK Avg.', 'Opp. TOV Avg.', 'Opp. PF Avg.', 'Opp. +/- Avg.']

# constants
THOUSAND = 10**3
MILLION = 10**6

def main():
    # get data
    nba = nba2017



    # # ----------------------------- Testing ------------------------------ #
    # for team_num in range(len(teams_str)):
    #     # team data
    #     team = getTeamGames(nba, teams_str[team_num])
    #     team = addDefensiveStats(team)
    #     team = addStatAverages(team)

    #     testing_inputs = numpy.array([team[input_categories].iloc[81].values])
    #     weight = weights[weights['team'] == teams_str[team_num]][input_categories].T
    #     output = sigmoid(numpy.dot(testing_inputs, weight))[0]
    #     actual_output = team['W/L'].iloc[81]

    
    
    # ----------------------------- Training ----------------------------- #
    # team data
    team_num = 3
    team = getTeamGames(nba, teams_str[team_num])
    team = addDefensiveStats(team)
    team = addStatAverages(team)
    print('\n', teams_str[team_num])

    training_inputs = numpy.array(team[input_categories].iloc[10:81].values)
    training_outputs = numpy.array([team['W/L'].iloc[10:81].values])

    # translate wins and losses into 1's and 0's 
    training_outputs[training_outputs == 'W'] = 1
    training_outputs[training_outputs == 'L'] = 0
    training_outputs = numpy.array(training_outputs).T

    numpy.random.seed(1)
    random_weight = numpy.random.random((len(training_inputs[0]), 1))
    syn_weights = 2 * random_weight - 1
    last_weights = numpy.array([numpy.array([487562.27382121776,-184240.30894955387,-620812.5878834525,-477246.4795795577,197238.11262495082,-85098.47307518324,-43074.69464663881,658804.810767647,-453229.11675516336,-262012.8135326605,379740.64419553406,-284607.0528064795,95132.79150466964,195244.61413523505,1160918.016711736,-307954.7094678382,-104447.11675371887,451703.6268669787,-89919.26655263749,357749.0503220181,-16477.50227554932,367780.2911298117,-792261.9318905178,233451.39433303758,-496679.95708932576,-43660.669255827495,157254.0221447926,-40771.92403723044,-21114.418540193026,-914117.4909215842,833781.554292566,-80336.04723500759,-336082.7275695222,-209257.46063896792,-334903.0998459643,441972.0518097461,149207.80760557402,-333795.39166182873])]).T

    widgets = [Bar(marker='=',left='[',right=']'), ' ', Percentage(), ' ', ETA(), ' ', FileTransferSpeed()]
    range_val = MILLION

    new_weights = trainTeam(training_inputs, training_outputs, last_weights)

    # train according to accuracy
    # while(accuracy <= 85):
    #     input_layer = training_inputs
    #     outputs = sigmoid(numpy.dot(input_layer, syn_weights))
    #     error = training_outputs - outputs
    #     adjustments = error * sigmoid_prime(outputs)
    #     syn_weights = syn_weights + numpy.dot(input_layer.T, adjustments)
    #     accuracy = float(getAccuracyPercentage(outputs, training_outputs))
    #     iterations = iterations + 1
    
    # train according to x iterations
    # while (more):
        # progress_bar = ProgressBar(widgets=widgets, maxval=range_val)
        # progress_bar.start()
        # for i in range(range_val):
        #     input_layer = training_inputs
        #     outputs = sigmoid(numpy.dot(input_layer, syn_weights))
        #     error = training_outputs - outputs
        #     adjustments = error * sigmoid_prime(outputs)
        #     syn_weights = syn_weights + numpy.dot(input_layer.T, adjustments)
        #     progress_bar.update(i)
        # progress_bar.finish()
        # print('Accuracy: ' + getAccuracyPercentage(outputs, training_outputs) + '%\n')
        # m = input('Would you like to run another loop? (y/n) ')
        # if m.lower() == 'n' or m.lower() == 'no':
        #     more = False



    # ------------------------------ Output ------------------------------ #
    # print('\n', syn_weights, '\n\n')
    print('\n', new_weights, '\n\n')
    # print('Accuracy: ' + getAccuracyPercentage(outputs, training_outputs) + '%\n')
    # print(iterations)


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
    print(iterations)
    return syn_weights

# Activation Functions
def sigmoid(x):
    return 1 / (1 + numpy.exp(-(x.astype(float))))
def sigmoid_prime(x):
    return sigmoid(x.astype(float)) * (1 - sigmoid(x.astype(float)))
    # return x.astype(float) * (1 - x.astype(float))

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
        return(data[nba2017['W/L'].str.contains('W')])
    else:
        return(data[nba2017['W/L'].str.contains('W')])[criteria]
def getLosses(data, criteria=None):
    if criteria is None:
        return(data[nba2017['W/L'].str.contains('L')])
    else:
        return(data[nba2017['W/L'].str.contains('L')])[criteria]
def getHome(data, criteria=None):
    if criteria is None:
        return data[data['MATCHUP'].str.contains('vs.')]
    else:
        return(data[nba2017['MATCHUP'].str.contains('vs.')])[criteria]
def getAway(data, criteria=None):
    if criteria is None:
        return data[data['MATCHUP'].str.contains('@')]
    else:
        return data[data['MATCHUP'].str.contains('@')][criteria]
def getTeamGames(data, team_name, criteria=None):
    if criteria is None:
        return(data[nba2017['TEAM'].str.contains(team_name)])
    else:
        return(data[nba2017['TEAM'].str.contains(team_name)])[criteria]
def getAverages(data):
    results = {}
    ignore_categories = ['TEAM', 'DATE', 'MATCHUP', 'W/L', 'MIN']
    for stat in data.keys().tolist():
        if stat not in ignore_categories and stat[-4:] != 'Avg.':
            results[stat] = data[stat].mean()
    return results
def getOpponentData(team, opponent, date, data=nba2017):
    return(nba2017[nba2017['MATCHUP'].str.contains(team) & nba2017['MATCHUP'].str.contains(opponent) & nba2017['DATE'].str.contains(date) & nba2017['TEAM'].str.contains(opponent)])

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

# KNN
def kmeans():
    pass
    # #     //------------------------------------------------- Main K Means --------------------------------------------------\\ #
    # 
    # # -------------------- Data -------------------- #
    #
    # training_categories = original_categories
    # testing_categories = []
    # ignore_categories = ['TEAM', 'DATE', 'MATCHUP', 'W/L', 'MIN', 'PTS', '+/-', 'FGM', 'FG%', '3PM', '3P%', 'FTM', 'FT%']
    #
    # # adds averages and opposing stats to criteria
    # average_criteria = []
    # defense_criteria = []
    # for x in ignore_categories:
    #     training_categories.remove(x)
    # for x in training_categories:
    #     defense_criteria.append('Opp. ' + x)
    # for x in defense_criteria:
    #     training_categories.append(x)
    # for x in training_categories:
    #     average_criteria.append(x + ' Avg.')
    # for x in average_criteria:
    #     training_categories.append(x) 
    #     testing_categories.append(x)
    #
    #
    # # --------------------- ML --------------------- #
    # train_data = team[testing_categories].iloc[:30]
    # train_labels = team['W/L'].iloc[:30]
    #
    # test_data = team[testing_categories].iloc[30:60]
    # test_labels = team['W/L'].iloc[30:]
    #
    # model = KNeighborsClassifier(n_neighbors=3)
    # model.fit(train_data, train_labels)
    # 
    # predictions = model.predict(test_data)
    # 
    #
    # # ------------------ Accuracy ------------------ #
    # right = 0
    # total = 0
    # for i in range(len(predictions)):
    #     # # print all predictions
    #     # print(predictions[i])
    #     # print(test_labels.tolist()[i])
    #     # print('----------------------')
    #     if predictions[i] == test_labels.tolist()[i]:
    #         right = right + 1
    #     total = total + 1
    #
    #
    # # ------------------- Graph -------------------- #
    # predictions = translatePredictions(predictions)
    # plt.title(team['TEAM'].iloc[0])
    # plt.xlabel('Point Spread')
    # plt.ylabel('FG%')
    # plt.legend(loc='upper left')
    # plt.scatter(team['+/-'].iloc[30:60], team['FG% Avg.'].iloc[30:60], c=predictions)
    # plt.show()
    # 
    # # //---------------------------------------------------- K Means ----------------------------------------------------\\ #

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
