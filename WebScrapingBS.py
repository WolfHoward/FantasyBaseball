
# coding: utf-8

# ## Using Beautiful Soup to web scrape MLB statistics ##
# I will be adapting the excellent tutorial at [Nylon Calculus](http://nyloncalculus.com/2015/09/07/nylon-calculus-101-data-scraping-with-python/) to scrape stats from [baseball-reference.com](http://www.baseball-reference.com/leagues/MLB/bat.shtml). I'll start with annual stats for the entire league, and in this section I'll focus specifically on batting. I will repeat the procedure for pitching and then dive into individual player statistics.
# All of this is an exercise in web scraping and data cleaning as I prepare for my first year in a Fantasy Baseball league. It will likely be obvious I have no idea what I'm talking about at some points.

# We begin by importing the necesary modules and loading the webpage with our data into the magic that is BeautifulSoup.

# In[1]:

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import datetime


def int_if_possible(string):
    try:
        return(int(string))
    except:
        return(string)


def team_batting_calc(teams=['CHW'], start=2016, end=2016):

    batting_df = pd.DataFrame()
    url_template = "http://www.baseball-reference.com/teams/{team}/{year}.shtml"

    for tm in teams:
        for year in range(start, end + 1):
            team = tm

            url = url_template.format(team=team, year=str(year))
            html = urlopen(url)

            soup = BeautifulSoup(html, 'lxml')
            headers = soup.findAll('tr')[0].findAll('th')

            column_headers = [th.getText() for th in headers[1:]]
            data_rows = soup.findAll('tbody')[0].findAll('tr')
            data_rows[0].findAll('td')[2].getText()

            player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                           for i in range(len(data_rows))]
            player_data= int_if_possible(player_data)
            player_df = pd.DataFrame(player_data, columns = column_headers)
            player_df = player_df[player_df.Pos.notnull()]

            player_df.insert(0, 'Year', year)
            batting_df = batting_df.append(player_df, ignore_index=True)
            #print(batting_df)

    for i in batting_df.columns:
        batting_df[i] = pd.to_numeric(batting_df[i], errors='ignore')

    column_headers.insert(0,'Year')
    batting_df = batting_df.reindex_axis(column_headers, axis = 1)

    current_date = datetime.date.today()
    writer = pd.ExcelWriter('br-data-by-team.xlsx')
    batting_df.to_excel(writer,'{}'.format(current_date))
    writer.save()

    return(batting_df)



stl = team_batting_calc(teams=['STL'],start=2015, end = 2016)
stl['Year']
