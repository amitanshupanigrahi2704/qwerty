#Dataset Link: https://www.kaggle.com/datasets/hugomathien/soccer/data
# Source Code:
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, roc_curve, auc,
    recall_score, precision_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# Database connection
db_path = '/Users/amitanshupanigrahi/Downloads/database.sqlite'
conn = sqlite3.connect(db_path)

# Match data query
match_query = """
SELECT match_api_id, date, home_team_api_id, away_team_api_id, 
home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11,
away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 
FROM Match;
"""
match_df = pd.read_sql(match_query, conn)
match_df['date'] = pd.to_datetime(match_df['date'], errors='coerce')

# Player attributes query
player_attributes_query = """
SELECT player_api_id, date, overall_rating, crossing, finishing, heading_accuracy, 
short_passing, dribbling, ball_control, acceleration, sprint_speed, agility, 
reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, 
interceptions, positioning, vision, penalties, marking, standing_tackle, sliding_tackle 
FROM Player_Attributes;
"""
player_attributes_df = pd.read_sql(player_attributes_query, conn)
player_attributes_df['date'] = pd.to_datetime(player_attributes_df['date'], errors='coerce')

# Close connection
conn.close()

# Reshape match data
home_players = match_df.melt(
    id_vars=['match_api_id', 'date', 'home_team_api_id', 'away_team_api_id'],
    value_vars=[f'home_player_{i}' for i in range(1, 12)],
    var_name='home_player_num', value_name='player_api_id'
).drop(columns='home_player_num')

away_players = match_df.melt(
    id_vars=['match_api_id', 'date', 'home_team_api_id', 'away_team_api_id'],
    value_vars=[f'away_player_{i}' for i in range(1, 12)],
    var_name='away_player_num', value_name='player_api_id'
).drop(columns='away_player_num')

home_players['team'] = 'home'
away_players['team'] = 'away'
players_long_df = pd.concat([home_players, away_players], ignore_index=True)

# Merge with player attributes
merged_df = players_long_df.merge(
    player_attributes_df, on='player_api_id', suffixes=('', '_player'), how='left'
)

# Aggregation
aggregated_df = merged_df.groupby(['match_api_id', 'team', 'home_team_api_id', 'away_team_api_id']).mean().reset_index()

# Goal difference and results
aggregated_df['goal_difference'] = aggregated_df['home_team_goal'] - aggregated_df['away_team_goal']
aggregated_df['match_result'] = aggregated_df['goal_difference'].apply(
    lambda x: 'win' if x > 0 else ('draw' if x == 0 else 'lose')
)

# Encoding results
result_mapping = {'win': 1, 'draw': 0, 'lose': -1}
aggregated_df['match_result_encoded'] = aggregated_df['match_result'].map(result_mapping)

# Visualization: Goal Difference Distribution
sns.histplot(aggregated_df['goal_difference'], kde=True)
plt.title('Goal Difference Distribution')
plt.xlabel('Goal Difference')
plt.ylabel('Frequency')
plt.show()

# Pair Plot
sns.pairplot(aggregated_df[['goal_difference', 'crossing', 'dribbling', 'stamina']])
plt.show()

# Initial Ratings and Updates (PI/ELO) ...

# Prepare data for classification
X_elo = aggregated_df[['home_team_elo_rating', 'away_team_elo_rating']]
X_pi = aggregated_df[['home_team_pi_rating', 'away_team_pi_rating']]
y = aggregated_df['match_result'].map({'win': 1, 'draw': 0, 'lose': -1})

X_train_elo, X_test_elo, y_train, y_test = train_test_split(X_elo, y, test_size=0.2, random_state=42)
X_train_pi, X_test_pi, _, _ = train_test_split(X_pi, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP Classifier": MLPClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4, l2_leaf_reg=5,
        bootstrap_type='Bernoulli', subsample=0.8, early_stopping_rounds=100,
        eval_metric='Accuracy', random_state=42
    )
}

# Training and evaluation logic here...
