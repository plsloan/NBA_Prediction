import numpy
import pandas
import warnings

import basketball_reference_scraper
from knn_learner import KNN_Learner
from train_model import getTeamGames, addDefensiveStats, addStatAverages
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
nba = pandas.read_csv('Data/NBA_'+ season_years + '_Data.csv')      # local data
# nba = basketball_reference_scraper.main(season_years)               # current data (scraped)
weights = pandas.read_csv('Weights/weights_'+ season_years + '.csv')

# categories
original_categories = ['TEAM','DATE','MATCHUP','W/L','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','PF','+/-']
input_categories    = ['PTS Avg.', 'FGM Avg.', 'FGA Avg.', 'FG% Avg.', '3PM Avg.', '3PA Avg.', '3P% Avg.', 'FTM Avg.', 'FTA Avg.', 'FT% Avg.', 'OREB Avg.', 'DREB Avg.', 'REB Avg.', 'AST Avg.', 'STL Avg.', 'BLK Avg.', 'TOV Avg.', 'PF Avg.', '+/- Avg.', 'Opp. PTS Avg.', 'Opp. FGM Avg.', 'Opp. FGA Avg.', 'Opp. FG% Avg.', 'Opp. 3PM Avg.', 'Opp. 3PA Avg.', 'Opp. 3P% Avg.', 'Opp. FTM Avg.', 'Opp. FTA Avg.', 'Opp. FT% Avg.', 'Opp. OREB Avg.', 'Opp. DREB Avg.', 'Opp. REB Avg.', 'Opp. AST Avg.', 'Opp. STL Avg.', 'Opp. BLK Avg.', 'Opp. TOV Avg.', 'Opp. PF Avg.', 'Opp. +/- Avg.']

# constants
THOUSAND = 10**3
MILLION = 10**6

def main():
    learners = []
    print('\n//------------------- Training ------------------\\\\')

    for team_str in teams_str:
        learner = KNN_Learner(k_num=3)
        team = getTeamGames(nba, team_str)
        team = addDefensiveStats(team)
        team = addStatAverages(team)

        training_inputs = numpy.array(team[input_categories].values)[10:50]
        training_outputs = numpy.array([team['PTS'].values])[0][10:50]
        testing_input = numpy.array(team[input_categories].values)[51:63]
        testing_pts = numpy.array([team['PTS'].values])[0][51:63]

        learner.train(training_inputs, training_outputs)
        y = learner.predict(testing_input)

        print('\n', team_str)
        print(y)
        print(testing_pts)
        learners.append(learner)
        
if __name__ == "__main__":
    main()