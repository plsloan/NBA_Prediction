import pandas
from requests import get
from datetime import datetime
from bs4 import BeautifulSoup, Comment


teams_str = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
teams_val = [  0  ,   1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ,   8  ,   9  ,   10 ,   11 ,   12 ,   13 ,   14 ,   15 ,   16 ,   17 ,   18 ,   19 ,   20 ,   21 ,   22 ,   23 ,   24 ,   25 ,   26 ,   27 ,   28 ,   29 ]

def main():
    for team_num in teams_val:
        print(team_num)
        season_stats = get_season_stats(teams_str[team_num])
        get_game_stats(teams_str[team_num])
        print()
    

def get_season_stats(team_str, year=2019):
    year = str(year)
    if team_str == 'CHA':
        url = 'https://www.basketball-reference.com/teams/' + 'CHO' + '/' + year + '.html'
    elif team_str == 'BKN':
        url = 'https://www.basketball-reference.com/teams/' + 'BRK' + '/' + year + '.html'
    elif team_str == 'PHX':
        url = 'https://www.basketball-reference.com/teams/' + 'PHO' + '/' + year + '.html'
    else:
        url = 'https://www.basketball-reference.com/teams/' + team_str.upper() + '/' + year + '.html'
    res = get(url, headers={"User-Agent":"Mozilla/5.0"})
    html_soup = BeautifulSoup(res.text, 'lxml')
    table_rows = []
    # if html_soup.find('table', id='games') == None:
    for comment in html_soup.find_all(string=lambda text:isinstance(text,Comment)):
        data = BeautifulSoup(comment,"lxml")
        for items in data.select("table#team_and_opponent tr"):
            tds = [item.get_text(strip=True) for item in items.select("th,td")]
            table_rows.append(tds)
    # else:
    #     table_rows = html_soup.find('table', id='games').find('tbody').find_all('tr')
    categories = table_rows[0][1:]
    totals = table_rows[1][1:]
    averages = table_rows[2][1:]
    league_ranks = table_rows[3][1:]
    year_to_year_change = table_rows[4][1:]
    opp_totals = table_rows[5][1:]
    opp_averages = table_rows[6][1:]
    opp_ranks = table_rows[7][1:]
    opp_year_to_year_change = table_rows[8][1:]
    return_array = [categories, totals, averages, league_ranks, year_to_year_change, opp_totals, opp_averages, opp_ranks, opp_year_to_year_change]
    return return_array
def get_game_stats(team_str_fun, year=2019):
    year = str(year)
    if team_str_fun == 'CHA':
        url = 'https://www.basketball-reference.com/teams/' + 'CHO' + '/' + year + '_games.html'
    elif team_str_fun == 'BKN':
        url = 'https://www.basketball-reference.com/teams/' + 'BRK' + '/' + year + '_games.html'
    elif team_str_fun == 'PHX':
        url = 'https://www.basketball-reference.com/teams/' + 'PHO' + '/' + year + '_games.html'
    else:
        url = 'https://www.basketball-reference.com/teams/' + team_str_fun + '/' + year + '_games.html'
    res = get(url, headers={"User-Agent":"Mozilla/5.0"})
    html_soup = BeautifulSoup(res.text, 'lxml')
    table = html_soup.find('table', id='games')
    body = table.find('tbody')
    rows = body.find_all('tr')
    stat_dict = {   'min': [], 'pts': [], 
                    'fgm': [], 'fga': [], 'fgp':[], 
                    'ftm': [], 'fta': [], 'ftp':[], 
                    '3pm': [], '3pa': [], '3pp':[], 
                    'oreb':[], 'dreb':[], 'reb':[],
                    'ast': [], 'stl': [], 'blk':[],
                    'tov': [], 'pf' : [], 'pm': []      }
    for row in rows:
        row_id = row.find('th').text
        row_data = row.find_all('td')   # date[0], box_score[2], home_away[3], opponent[4], win_loss[5], team_points[6], opp_points[7]
        if len(row_data) > 0 and datetime.now() > datetime.strptime(row_data[0].text[5:], '%b %d, %Y'):
            date = row_data[0].text
            box_score_url = 'https://www.basketball-reference.com' + row_data[3].find('a')['href']
            home_away = row_data[4].text
            opponent = row_data[5].text
            opp_str  = get_opponent_str(opponent)
            win_loss = row_data[6].text
            team_points = row_data[8].text
            opp_points = row_data[9].text
            if home_away == '':     # home
                matchup = team_str_fun + ' vs. ' + opp_str
            else:                   # away
                matchup = team_str_fun +  ' @ '  + opp_str
            res = get(box_score_url, headers={"User-Agent":"Mozilla/5.0"})
            html_soup = BeautifulSoup(res.text, 'lxml')
            if team_str_fun == 'CHA':
                team_table = html_soup.find('table', id='box_' + 'cho' + '_basic')
            elif team_str_fun == 'BKN':
                team_table = html_soup.find('table', id='box_' + 'brk' + '_basic')
            elif team_str_fun == 'PHX':
                team_table = html_soup.find('table', id='box_' + 'pho' + '_basic')
            else:
                team_table = html_soup.find('table', id='box_' + team_str_fun.lower() + '_basic')
            team_stats = team_table.find('tfoot').find('tr')
            team_data  = team_stats.find_all('td')
            if opp_str == 'CHA':
                opp_table  = html_soup.find('table', id='box_' + 'cho' + '_basic')
            elif opp_str == 'BKN':
                opp_table  = html_soup.find('table', id='box_' + 'brk' + '_basic')
            elif opp_str == 'PHX':
                opp_table  = html_soup.find('table', id='box_' + 'pho' + '_basic')
            else:
                opp_table  = html_soup.find('table', id='box_' + opp_str.lower() + '_basic')
            opp_stats = opp_table.find('tfoot').find('tr')
            opp_data = opp_stats.find_all('td')
            print(team_points, opp_points, win_loss)
            

def parse_date(str_date):
    split_date = str_date.split(',')[1:]
    month = convert_month(split_date[0].strip().split(' ')[0])
    day = split_date[0].strip().split(' ')[1]
    year = split_date[1].strip()
    return year, month, day
def convert_month(str_month):
    if str_month.lower() == 'jan':
        return str(1)
    elif str_month.lower() == 'feb':
        return str(2)
    elif str_month.lower() == 'mar':
        return str(3)
    elif str_month.lower() == 'apr':
        return str(4)
    elif str_month.lower() == 'may':
        return str(5)
    elif str_month.lower() == 'jun':
        return str(6)
    elif str_month.lower() == 'jul':
        return str(7)
    elif str_month.lower() == 'aug':
        return str(8)
    elif str_month.lower() == 'sep':
        return str(9)
    elif str_month.lower() == 'oct':
        return str(10)
    elif str_month.lower() == 'nov':
        return str(11)
    elif str_month.lower() == 'dec':
        return str(12)
    else:
        return None
def get_opponent_str(opponent):
    if opponent[:3].upper() == 'ATL':
        return teams_str[0]
    elif opponent[:3].upper() == 'BOS':
        return teams_str[1]
    elif opponent.upper() == 'BROOKLYN NETS':
        return teams_str[2]
    elif opponent[:3].upper() == 'CHA':
        return teams_str[3]
    elif opponent[:3].upper() == 'CHI':
        return teams_str[4]
    elif opponent[:3].upper() == 'CLE':
        return teams_str[5]
    elif opponent[:3].upper() == 'DAL':
        return teams_str[6]
    elif opponent[:3].upper() == 'DEN':
        return teams_str[7]
    elif opponent[:3].upper() == 'DET':
        return teams_str[8]
    elif opponent[:6].upper() == 'GOLDEN':
        return teams_str[9]
    elif opponent[:3].upper() == 'HOU':
        return teams_str[10]
    elif opponent[:3].upper() == 'IND':
        return teams_str[11]
    elif opponent.upper() == 'LOS ANGELES CLIPPERS':
        return teams_str[12]
    elif opponent.upper() == 'LOS ANGELES LAKERS':
        return teams_str[13]
    elif opponent[:3].upper() == 'MEM':
        return teams_str[14]
    elif opponent[:3].upper() == 'MIA':
        return teams_str[15]
    elif opponent[:3].upper() == 'MIL':
        return teams_str[16]
    elif opponent[:3].upper() == 'MIN':
        return teams_str[17]
    elif opponent.upper() == 'NEW ORLEANS PELICANS':
        return teams_str[18]
    elif opponent.upper() == 'NEW YORK KNICKS':
        return teams_str[19]
    elif opponent.upper() == 'OKLAHOMA CITY THUNDER':
        return teams_str[20]
    elif opponent[:3].upper() == 'ORL':
        return teams_str[21]
    elif opponent[:3].upper() == 'PHI':
        return teams_str[22]
    elif opponent.upper() == 'PHOENIX SUNS':
        return teams_str[23]
    elif opponent[:3].upper() == 'POR':
        return teams_str[24]
    elif opponent.upper() == 'SACRAMENTO KINGS':
        return teams_str[25]
    elif opponent.upper() == 'SAN ANTONIO SPURS':
        return teams_str[26]
    elif opponent[:3].upper() == 'TOR':
        return teams_str[27]
    elif opponent[:3].upper() == 'UTA':
        return teams_str[28]
    elif opponent[:3].upper() == 'WAS':
        return teams_str[29]

if __name__ == '__main__':
    main()




#print [item["data-bin"] for item in bs.find_all() if "data-bin" in item.attrs]