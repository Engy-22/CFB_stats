# CFB_stats
College football spread predictor using Pandas and Scikit-Learn

Includes Python class for model creation. In order to create predictions, use the create_model function within the CFB class. The format of a correct call for the function includes the year range to use as a basis for developing model data, a prediction year to indicate which year's games the model is predicting, and a list of games included.

A proper call is shown below:
CFB(2010,2018).create_model(2019,[['Clemson','Alabama'],['LSU','Georgia']])

The resulting dataframe includes the away and home team, the lasso and ridge model predicted winners, and the spread produced by each model for the winning team, shown as a negative value to match betting lines. A csv file is also generated with the call, taking the name 'CFB_game_results.csv'.

All statistics are acqired from the site 'https://www.sports-reference.com/cfb/'. The 'Team Offense' and 'Team Defense' tables are used for each year and conference. Conference rankings are obtained from the 'Conference Summary' table, and the schedule is taken from the 'Schedule and Scores' tab.

Other details of the model are provided in the comments.
