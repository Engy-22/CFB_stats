# CFB_stats
College football spread predictor.

Includes Python class for model creation. In order to create predictions, use the create_model function within the CFB class. The format of a correct call for the function includes the year range to use as a basis for developing model data, a prediction year to indicate which year's games the model is predicting, and a list of games included.

A proper call is shown below:
CFB(2010,2018).create_model(2019,[['Clemson','Alabama'],['LSU','Georgia']])

The resulting dataframe includes the away and home team, the lasso and ridge model predicted winners, and the spread produced by each model for the winning team, shown as a negative value to match betting lines. A csv file is also generated with the call, taking the name 'CFB_game_results.csv'.

All statistics are acqired from a college football database (Sports Reference, LLC). The 'Team Offense' and 'Team Defense' tables are used for each year and conference. Conference rankings are obtained from the 'Conference Summary' table, the schedule is taken from the 'Schedule and Scores' tab, and the team ratings are taken from the 'School Ratings' table.

Other details of the model are provided in the comments. Updates on progress of the model are given for the generation of input values by year and the generation of prediction values by game number.

Edited as of November 6, 2019 to exclude year 2017 from model input, as ratings have been deleted for most teams.

Source:
"College Football Stats and History". Sports Reference, LLC. 2019. https://www.sports-reference.com/cfb/.
