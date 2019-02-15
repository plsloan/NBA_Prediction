from requests import get
from bs4 import BeautifulSoup, Comment


teams_str = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
teams_val = [  0  ,   1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ,   8  ,   9  ,   10 ,   11 ,   12 ,   13 ,   14 ,   15 ,   16 ,   17 ,   18 ,   19 ,   20 ,   21 ,   22 ,   23 ,   24 ,   25 ,   26 ,   27 ,   28 ,   29 ]

def main():
    team_num = 0
    season_stats = get_season_stats(teams_str[team_num])
    get_game_stats(teams_str[team_num])
    

def get_season_stats(team_str, year=2019):
    year = str(year)
    url = 'https://www.basketball-reference.com/teams/' + team_str + '/' + year + '.html'
    res = get(url, headers={"User-Agent":"Mozilla/5.0"})
    html_soup = BeautifulSoup(res.text, 'lxml')
    table_rows = []
    for comment in html_soup.find_all(string=lambda text:isinstance(text,Comment)):
        data = BeautifulSoup(comment,"lxml")
        for items in data.select("table#team_and_opponent tr"):
            tds = [item.get_text(strip=True) for item in items.select("th,td")]
            table_rows.append(tds)
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
def get_game_stats(team_str, year=2019):
    year = str(year)
    url = 'https://www.basketball-reference.com/teams/' + team_str + '/' + year + '_games.html'
    res = get(url, headers={"User-Agent":"Mozilla/5.0"})
    html_soup = BeautifulSoup(res.text, 'lxml')
    table = html_soup.find('table', id='games')
    body = table.find('tbody')
    rows = body.find_all('tr')
    for row in rows:
        row_id = row.find('th').text
        row_data = row.find_all('td')   # date[0], box_score[2], home_away[3], opponent[4], win_loss[5], team_points[6], opp_points[7]
        if len(row_data) > 0 and row_data[3].find('a'):   # if box score is found
            date = row_data[0].text
            box_score_url = 'https://www.basketball-reference.com' + row_data[3].find('a')['href']
            home_away = row_data[4].text
            opponent = row_data[5].text
            win_loss = row_data[6].text
            team_points = row_data[8].text
            opp_points = row_data[9].text

def parseDate(str_date):
    split_date = str_date.split(',')[1:]
    month = convertMonth(split_date[0].strip().split(' ')[0])
    day = split_date[0].strip().split(' ')[1]
    year = split_date[1].strip()
    return year, month, day
def convertMonth(str_month):
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
    
if __name__ == '__main__':
    main()