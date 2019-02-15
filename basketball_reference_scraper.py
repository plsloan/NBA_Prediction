from requests import get
from bs4 import BeautifulSoup, Comment


def main():
    season_stats = get_season_stats('BOS')
    for row in season_stats:
        print(row)


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




if __name__ == '__main__':
    main()