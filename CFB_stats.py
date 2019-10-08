import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
import numpy as np
from datetime import datetime


class CFB:
    """
    Uses machine learning algorithms to simulate college football DI-A matchups and corresponding spreads.
    Class takes two variables, year1 and year2, which correspond to the range of years used as a basis for
    model input data gathering. Data sourced from https://www.sports-reference.com/cfb/.
    """

    # Dictionary of current conferences and teams in conference for all D-1A, as of 2019 season. The teams
    # are listed in the format the model accepts.

    current_conf = \
        {
            'acc': ['Virginia Tech', 'Boston College', 'Georgia Tech', 'Florida State',
                    'Clemson', 'Pitt', 'Louisville', 'North Carolina',
                    'Syracuse', 'Virginia', 'Duke', 'Miami (FL)',
                    'North Carolina State', 'Wake Forest'],
            'american': ['UCF', 'Memphis', 'Tulsa', 'East Carolina', 'Navy',
                         'Cincinnati', 'SMU', 'Tulane', 'Connecticut',
                         'South Florida', 'Houston', 'Temple'],
            'big-12': ['Oklahoma', 'Kansas', 'Texas Christian', 'Texas Tech',
                       'Kansas State', 'Baylor', 'Oklahoma State', 'Texas', 'Iowa State',
                       'West Virginia'],
            'big-ten': ['Northwestern', 'Rutgers', 'Penn State', 'Minnesota', 'Nebraska',
                        'Michigan State', 'Wisconsin', 'Indiana', 'Michigan', 'Ohio State',
                        'Purdue', 'Illinois', 'Iowa', 'Maryland'],
            'cusa': ['Florida International', 'Old Dominion', 'Rice', 'Charlotte',
                     'Louisiana Tech', 'Marshall', 'Florida Atlantic', 'UTEP',
                     'Middle Tennessee State', 'UAB', 'North Texas',
                     'UTSA', 'Western Kentucky', 'Southern Mississippi'],
            'independent': ['Liberty', 'Army', 'Massachusetts', 'Notre Dame', 'Brigham Young', 'New Mexico State'],
            'mac': ['Eastern Michigan', 'Central Michigan', 'Miami (OH)', 'Kent State',
                    'Northern Illinois', 'Western Michigan', 'Buffalo', 'Ohio',
                    'Bowling Green State', 'Ball State', 'Akron', 'Toledo'],
            'mwc': ['Hawaii', 'Air Force', 'Boise State', 'Nevada', 'Colorado State',
                    'Fresno State', 'Wyoming', 'New Mexico', 'San Diego State',
                    'UNLV', 'San Jose State', 'Utah State'],
            'pac-12': ['USC', 'Washington', 'Washington State', 'Arizona',
                       'Stanford', 'Arizona State', 'UCLA', 'Oregon State', 'Oregon',
                       'Utah', 'California', 'Colorado'],
            'sec': ['Georgia', 'Kentucky', 'Alabama', 'LSU', 'Texas A&M',
                    'Arkansas', 'Florida', 'Mississippi State', 'South Carolina',
                    'Tennessee', 'Ole Miss', 'Vanderbilt', 'Auburn', 'Missouri'],
            'sun-belt': ['Texas State', 'Coastal Carolina', 'Troy', 'South Alabama',
                         'Arkansas State', 'Louisiana-Monroe', 'Georgia State', 'Louisiana',
                         'Appalachian State', 'Georgia Southern']
        }

    # All conferences existing from 2000 to 2019, with abbreviations used to convert from rank
    # dataframe to URL input.

    hist_conf = \
        {
            'Southeastern Conference': 'sec', 'Big East Conference': 'big-east',
            'Atlantic Coast Conference': 'acc', 'Big 12 Conference': 'big-12',
            'American Athletic Conference': 'american',
            'Pacific-10 Conference': 'pac-10', 'Pacific-12 Conference': 'pac-12',
            'Big Ten Conference': 'big-ten', 'Mountain West Conference': 'mwc',
            'Independent': 'independent', 'Western Athletic Conference': 'wac',
            'Conference USA': 'cusa', 'Mid-American Conference': 'mac',
            'Sun Belt Conference': 'sun-belt', 'Big West Conference': 'big-west'
        }

    def __init__(self, year1, year2):
        """Initializes class inputs, range of years for model input."""
        self.year1 = year1
        self.year2 = year2

    def CFB_stats(self, conf):
        """
        Coverts team offense and team defense datasets provided by Sports Reference into usable
        Pandas dataframes. Uses year1 as basis for data gathering.
        """

        # Convert PAC-12 to PAC-10 prior to 2011 for team offense. Then generate correct URL.
        if conf == 'pac-12' and self.year1 < 2011:
            url_off = 'https://www.sports-reference.com/cfb/conferences/pac-10/{}-team-offense.html'.format \
                (self.year1)
        else:
            url_off = 'https://www.sports-reference.com/cfb/conferences/{}/{}-team-offense.html'.format \
                (conf, self.year1)

        # Request data using URL, join 0 and 1 level columns labels and format properly, drop games and points
        # columns, and add 'Off' to beginning of column names to delineate offensive stats. Stats already
        # averaged per game, so did not average by game column.
        off = pd.read_html(url_off)[0].iloc[:, 1:]
        off.columns = [' '.join(col) for col in off.columns.values]
        for col in off.columns[:3]:
            off.rename(columns={col: col.split('0')[-1].strip()}, inplace=True)
        off.drop(columns=["G", 'Pts'], inplace=True)
        for col in off.columns[1:]:
            off.rename(columns={col: 'Off {}'.format(col)}, inplace=True)

        # Again convert PAC-12 to PAC-10 for years prior to 2011 and create URL.
        if conf == 'pac-12' and self.year1 < 2011:
            url_def = 'https://www.sports-reference.com/cfb/conferences/pac-10/{}-team-defense.html'.format \
                (self.year1)
        else:
            url_def = 'https://www.sports-reference.com/cfb/conferences/{}/{}-team-defense.html'.format \
                (conf, self.year1)

        # Perform same data cleaning as with team offense, and similarly add 'Def' before column name to
        # delineate defensive stats.
        defense = pd.read_html(url_def)[0].iloc[:, 1:]
        defense.columns = [' '.join(col) for col in defense.columns.values]
        for col in defense.columns[:3]:
            defense.rename(columns={col: col.split('0')[-1].strip()}, inplace=True)
        defense.drop(columns=["G", 'Pts'], inplace=True)
        for col in defense.columns[1:]:
            defense.rename(columns={col: 'Def {}'.format(col)}, inplace=True)

        # Merge offensive and defensive stats to form all stats for that year and conference, and fill
        # all na values with 0.
        all_stats = pd.merge(left=off, right=defense, on="School")
        all_stats = all_stats.fillna(0)
        return all_stats

    def schedule(self):
        """Generates CFB DI-A schedule from given year (year1) using Sports Reference."""

        # Format URL to retrieve schedule from year1, read url, and select relevent columns.
        # Unnamed column indicates '@' for away or is nan.
        url = 'https://www.sports-reference.com/cfb/years/{}-schedule.html'.format(self.year1)
        schedule = pd.read_html(url)[0]
        if self.year1 > 2012:
            schedule = schedule[['Date', 'Winner', 'Pts', 'Unnamed: 7', 'Loser', 'Pts.1']]
        else:
            schedule = schedule[['Date', 'Winner', 'Pts', 'Unnamed: 6', 'Loser', 'Pts.1']]

        # Name columns accordingly, and drop all non-numeric columns with labels.
        schedule.columns = ['Date', 'Winner', 'Winner Pts', 'Away/Home', 'Loser', 'Loser Pts']
        schedule = schedule[schedule['Winner'] != 'Winner'].copy()

        def strip_rank(team):
            """Strip ranking from team name."""
            if '(' in team:
                team = team.split(')')[-1]
            team = team.replace(u'\xa0', u'')
            return team

        # Strip rank from all teams.
        schedule['Winner'] = schedule['Winner'].apply(strip_rank)
        schedule['Loser'] = schedule['Loser'].apply(strip_rank)

        # Initialize Away and Home columns, as well as home spread.
        schedule['Away'] = schedule['Winner']
        schedule['Home'] = schedule['Loser']
        schedule['Home Spread'] = schedule['Winner Pts'].apply(float) - schedule['Loser Pts'].apply(float)

        # Chose home and away teams based on Away/Home column. Adjust teams if not in same order as
        # winner and loser.
        for i in range(len(schedule)):
            if schedule.iloc[i, 3] == '@':
                continue
            schedule.iloc[i, 6] = schedule.iloc[i, 4]
            schedule.iloc[i, 7] = schedule.iloc[i, 1]
            schedule.iloc[i, 8] = -schedule.iloc[i, 8]

        # Choose relevent columns, convert date to year, and drop games with empty values.
        schedule = schedule[['Date', 'Away', 'Home', 'Home Spread']]
        schedule = schedule.replace('', np.nan).dropna(axis=0)
        schedule['Year'] = [self.year1 for x in range(len(schedule))]
        schedule.drop(columns='Date', inplace=True)
        return schedule

    def ratings(self):
        """Obtains SRS ratings for each team in given year."""

        # Format URL, retrieve data, set columns, and filter out blank and heading rows.
        url = 'https://www.sports-reference.com/cfb/years/{}-ratings.html'.format(self.year1)
        ratings = pd.read_html(url)[0]
        ratings.columns = ratings.columns.get_level_values(1)
        ratings = ratings[ratings['SRS'] != 'SRS']
        ratings = ratings[ratings['W'] != 'Overall']
        ratings = ratings[['School', 'SRS']]

        # Strip school names, convert SRS to float value, and set school as index.
        ratings['School'] = ratings['School'].apply(lambda x: x.strip())
        ratings['SRS'] = ratings['SRS'].apply(float)
        ratings.set_index('School', inplace=True)

        # UNLV not in same format. Correct for format.
        ratings.rename(index={'Nevada-Las Vegas': 'UNLV'}, inplace=True)
        return ratings

    def rank(self):
        """Obtains ranking of conferences based on Sports Reference ranking. Used to obtain conference names."""

        # Request ranking for given year, isolate conference and ranking, and reformat conference into
        # abbrev format.
        url = 'https://www.sports-reference.com/cfb/years/{}.html'.format(self.year1)
        rank = pd.read_html(url)[0]
        rank.columns = rank.columns.get_level_values(1)
        rank = rank[['Rk', 'Conference']]
        rank['Conference'] = rank['Conference'].apply(lambda x: CFB.hist_conf[x])
        return rank

    def data_input(self):
        """Creates input data for model using given year range."""

        # Initialize dataframe for all game data
        all_games = pd.DataFrame()

        # Loop through all years in range. Obtain schedule, rating, and conference ranking for given year.
        for year in range(self.year1, self.year2 + 1):
            schedule = CFB(year, self.year2).schedule()
            ratings = CFB(year, self.year2).ratings()
            rank = CFB(year, self.year2).rank()

            # Inititialize conference stats and loop through conferences, obtaining stats for each conference
            # and ratings for each school. SAS ratings used to differentiate strength of schedule and opponent.
            all_conf_stats = pd.DataFrame()
            for conf in rank['Conference']:
                stats = CFB(year, self.year2).CFB_stats(conf)
                stats['Rating'] = stats['School'].apply(lambda x: ratings.loc[x, 'SRS'])

                # Concatenate stats from each conference.
                all_conf_stats = pd.concat([all_conf_stats, stats])

            # Compile away and home stats, and merge the two. Indicate when given year's data is compiled.
            away_stats = pd.merge(schedule, all_conf_stats, left_on='Away', right_on='School')
            total_stats = pd.merge(away_stats, all_conf_stats, left_on='Home', right_on='School')
            all_games = pd.concat([all_games, total_stats])
            print('Data from {} compiled'.format(year))

        # Rename columns of dataframe to include home and away. Drop indications of school and year to
        # anonymize data.
        for col in all_games.columns:
            if col[-1] == 'x':
                all_games.rename(columns={col: 'Away {}'.format(col.split('_')[0])}, inplace=True)
            elif col[-1] == 'y':
                all_games.rename(columns={col: 'Home {}'.format(col.split('_')[0])}, inplace=True)
        all_games.drop(columns=['Home Away', 'Home', 'Away School', 'Home School', 'Year'], inplace=True)
        return all_games

    def pred_input(self, pred_year, games):
        """
        Creates data for models to be used as basis for prediction.
        Based on the prediction games and the matchups. Takes list of lists as input for games.
        """

        # Initialize dataframe for storing current year stats on teams playing. Also obtain ratings and
        # initialize number of game to update user on progress.
        all_games = pd.DataFrame()
        ratings = CFB(pred_year, self.year2).ratings()
        game_num = 1

        # Loop through each game, first determining away and home conference stats.
        for game in games:
            away_conf = [k for k, v in CFB.current_conf.items() if game[0] in v][0]
            home_conf = [k for k, v in CFB.current_conf.items() if game[1] in v][0]
            away_conf_stats = CFB(pred_year, self.year2).CFB_stats(away_conf)
            home_conf_stats = CFB(pred_year, self.year2).CFB_stats(home_conf)

            # Determine stats of teams by extracting these teams from their respective conference.
            # Format stats using home and away. Detemine rating for each team.
            away_stats = away_conf_stats[away_conf_stats['School'] == game[0]] \
                .copy().reset_index().drop(columns='index')
            away_stats['Rating'] = away_stats['School'].apply(lambda x: ratings.loc[x, 'SRS'])
            for col in away_stats.columns:
                away_stats.rename(columns={col: 'Away {}'.format(col)}, inplace=True)
            home_stats = home_conf_stats[home_conf_stats['School'] == game[1]] \
                .copy().reset_index().drop(columns='index')
            home_stats['Rating'] = home_stats['School'].apply(lambda x: ratings.loc[x, 'SRS'])
            for col in home_stats.columns:
                home_stats.rename(columns={col: 'Home {}'.format(col)}, inplace=True)

            # Concatenate home and awat stats and drop school names, so that data matches model input.
            total = pd.concat([away_stats, home_stats], axis=1)
            total.drop(columns=['Away School', 'Home School'], inplace=True)

            # Concatenate game data with all games and indicate to user that game is compiled.
            all_games = pd.concat([all_games, total])
            print('Game {} compiled'.format(game_num))
            game_num += 1
        return all_games

    def create_model(self, pred_year, games):
        """
        Generates spread predictions based on model input in given year range,
        year of prediction, and games to be predicted.
        """
        # Check for proper ranges of years given.
        if self.year1 < 2000:
            return 'Data not available before 2000.'
        elif self.year2 <= self.year1:
            return 'Year 2 must be greater than year 1.'
        elif self.year2 >= pred_year:
            return 'Year 2 must be less than prediction year.'
        elif pred_year > datetime.now().year:
            return 'Prediction year must be less than or equal to current year.'

        # Determine if all games are in DI-A, and are in proper format.
        # Refer to current_conf for teams available.
        for game in games:
            for team in game:
                if team not in [x for k, v in CFB.current_conf.items() for x in v]:
                    return 'Either not all D1-A teams or not in proper format.'

        # Generate input values for model. Set X as everything excluding spread, and y as spread.
        input_values = CFB(self.year1, self.year2).data_input()
        X = input_values.iloc[:, 1:]
        y = input_values['Home Spread']

        # Generate models, with 5 folds, set max_iter to 10000 for lasso, and fit to data.
        lasso_mod = LassoCV(cv=5, max_iter=10000).fit(X, y)
        ridge_mod = RidgeCV(cv=5).fit(X, y)

        # Generate values for generating predictions, and create predictions.
        pred_values = CFB(self.year1, self.year2).pred_input(pred_year, games)
        lasso_pred = lasso_mod.predict(pred_values)
        ridge_pred = ridge_mod.predict(pred_values)

        # Create result dictionary, indicating home and away teams, predicted winners, and spread.
        results = {'Away': [x[0] for x in games], 'Home': [x[1] for x in games],
                   'Lasso Predicted Winner': [games[i][0] if lasso_pred[i] > 0 else games[i][1]
                                              for i in range(len(games))],
                   'Ridge Predicted Winner': [games[i][0] if ridge_pred[i] > 0 else games[i][1]
                                              for i in range(len(games))],
                   'Lasso Spread': [-abs(round(x, 1)) for x in lasso_pred],
                   'Ridge Spread': [-abs(round(x, 1)) for x in ridge_pred]}

        # Create dataframe based on dictionary, create index, and save as csv.
        results = pd.DataFrame(results)
        index = pd.Index(['Game {}'.format(num) for num in range(1, len(games) + 1)])
        results.index = index
        results.to_csv('CFB_games_results.csv')
        return results