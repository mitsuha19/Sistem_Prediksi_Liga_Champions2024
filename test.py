import logging
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load datasets
try:
    teams_data = pd.read_csv('archive/teams_data.csv')
    players_data = pd.read_csv('archive/DAY_4/players_data.csv')
    attacking_data = pd.read_csv('archive/DAY_4/attacking_data.csv').rename(columns={'player_id': 'id_player'})
    attempts_data = pd.read_csv('archive/DAY_4/attempts_data.csv').rename(columns={'em': 'id_player'})
    defending_data = pd.read_csv('archive/DAY_4/defending_data.csv')
    disciplinary_data = pd.read_csv('archive/DAY_4/disciplinary_data.csv')
    distribution_data = pd.read_csv('archive/DAY_4/distribution_data.csv')
    # goalkeeping_data = pd.read_csv('archive/DAY_4/goalkeeping_data.csv')  # Removed
    goals_data = pd.read_csv('archive/DAY_4/goals_data.csv')
    key_stats_data = pd.read_csv('archive/DAY_4/key_stats_data.csv')
except Exception as e:
    logger.error(f"Error loading CSV files: {e}")
    raise e

# Merge all player-related data
try:
    player_data = (
        attacking_data
        .merge(attempts_data, on='id_player', how='left')
        .merge(defending_data, on='id_player', how='left')
        .merge(disciplinary_data, on='id_player', how='left')
        .merge(distribution_data, on='id_player', how='left')
        # .merge(goalkeeping_data, on='id_player', how='left')  # Removed
        .merge(goals_data, on='id_player', how='left')
        .merge(key_stats_data, on='id_player', how='left')
        .merge(players_data, on='id_player', how='left')
    )
except Exception as e:
    logger.error(f"Error merging player data: {e}")
    raise e

# Merge with team data
try:
    merged_data = player_data.merge(teams_data, left_on='id_team', right_on='team_id', how='left')
    merged_data.fillna(0, inplace=True)
except Exception as e:
    logger.error(f"Error merging with team data: {e}")
    raise e

# Determine overall team probability
features = ['assists', 'total_attempts', 'tackles_won', 'distance_covered(km/h)', 'goals', 'passes_completed']
X = merged_data.groupby('team_id')[features].mean().reset_index()

def is_forward(player):
    pos = str(player['field_position']).lower()
    return "forward" in pos or "striker" in pos or "winger" in pos

def is_midfielder(player):
    pos = str(player['field_position']).lower()
    return "midfielder" in pos or "midfield" in pos

def is_defender(player):
    pos = str(player['field_position']).lower()
    return "defender" in pos or "back" in pos

def is_goalkeeper(player):
    pos = str(player['field_position']).lower()
    return "goalkeeper" in pos or "keeper" in pos or "gk" in pos or "goalie" in pos

def filter_players_by_role(team_players, role):
    if role == 'scorer':
        filtered = team_players[team_players.apply(is_forward, axis=1)]
        if filtered.empty:
            filtered = team_players
        return filtered
    elif role == 'assister':
        filtered = team_players[(team_players.apply(is_forward, axis=1)) | (team_players.apply(is_midfielder, axis=1))]
        if filtered.empty:
            filtered = team_players
        return filtered
    elif role == 'offside':
        filtered = team_players[team_players.apply(is_forward, axis=1)]
        if filtered.empty:
            filtered = team_players
        return filtered
    elif role == 'save':
        filtered = team_players[team_players.apply(is_goalkeeper, axis=1)]
        return filtered
    elif role == 'card':
        return team_players
    else:
        return team_players

def weighted_random_player(team_id, role='scorer'):
    team_players = merged_data[merged_data['team_id'] == team_id]
    team_players = filter_players_by_role(team_players, role)
    if team_players.empty:
        # Fallback
        team_players = merged_data[merged_data['team_id'] == team_id]
        if team_players.empty:
            logger.warning(f"No players found for team ID {team_id}")
            return None

    if role == 'scorer':
        weights = team_players['goals'] + (team_players['total_attempts'] * 0.5) + 1
    elif role == 'assister':
        weights = team_players['assists'] + (team_players['passes_completed'] * 0.01) + 1
    elif role == 'card':
        base = (team_players['fouls_committed'] + team_players['yellow_cards']*2 + team_players['red_cards']*5) + 1
        role_boost = team_players.apply(lambda p: 2 if is_defender(p) or is_midfielder(p) else 1, axis=1)
        weights = base * role_boost
    elif role == 'save':
        if 'saves' in team_players.columns:
            weights = team_players['saves'] + 1
        weights = team_players['passes_completed'] * 0.1 + 1
    elif role == 'offside':
        weights = team_players['offsides'] + 1
    else:
        weights = np.ones(len(team_players))

    weights = weights.fillna(1) + 0.0001
    if len(team_players) == 0:
        return None
    try:
        selected_player = team_players.sample(weights=weights, replace=False, n=1).iloc[0]
        return selected_player
    except Exception as e:
        logger.error(f"Error selecting player: {e}")
        return None

def substitution_event(chosen_team, minute):
    if minute <= 45:
        return None
    team_players = merged_data[merged_data['team_id'] == chosen_team]
    non_gk_players = team_players[~team_players.apply(is_goalkeeper, axis=1)]
    if len(non_gk_players) < 2:
        available_players = team_players
    else:
        available_players = non_gk_players

    positions = available_players['field_position'].unique()
    if len(positions) == 0:
        return None
    chosen_position = random.choice(positions)
    pos_players = available_players[available_players['field_position'] == chosen_position]
    if len(pos_players) < 2:
        pos_players = available_players
    if len(pos_players) < 2:
        return None

    out_player = pos_players.sample(1).iloc[0]
    in_candidates = pos_players[pos_players['id_player'] != out_player['id_player']]
    if in_candidates.empty:
        in_candidates = available_players[available_players['id_player'] != out_player['id_player']]

    if in_candidates.empty:
        return None
    in_player = in_candidates.sample(1).iloc[0]

    return {
        'minute': minute,
        'event': 'Substitution',
        'team': out_player['team'],
        'player': f"{out_player['player_name']} -> {in_player['player_name']}",
        'photo': in_player['player_image'],
        'assist_text': "",
        'assist_photo': out_player['player_image'] 
    }

@app.route('/')
def home():
    teams = teams_data[['team_id', 'team', 'logo']].drop_duplicates().to_dict(orient='records')
    return render_template('home.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        team1_id = int(request.form['team1'])
        team2_id = int(request.form['team2'])
        logger.debug(f"Predicting match between Team {team1_id} and Team {team2_id}")

        if team1_id == team2_id:
            logger.error("A team cannot play against itself.")
            return "Error: A team cannot play against itself."

        team1_stats_row = X[X['team_id'] == team1_id]
        team2_stats_row = X[X['team_id'] == team2_id]

        if team1_stats_row.empty:
            logger.error(f"Team ID {team1_id} not found in statistics.")
            return f"Error: Team ID {team1_id} not found in statistics."

        if team2_stats_row.empty:
            logger.error(f"Team ID {team2_id} not found in statistics.")
            return f"Error: Team ID {team2_id} not found in statistics."

        team1_stats = team1_stats_row.iloc[0].mean()
        team2_stats = team2_stats_row.iloc[0].mean()

        total_stats = team1_stats + team2_stats
        team1_prob = team1_stats / total_stats if total_stats != 0 else 0.5
        team2_prob = team2_stats / total_stats if total_stats != 0 else 0.5

        # Adjust goal probabilities to favor higher-ranked team
        if team1_prob >= team2_prob:
            higher_prob = team1_prob
            lower_prob = team2_prob
        else:
            higher_prob = team2_prob
            lower_prob = team1_prob

        score_range = range(0, 5)
        team1_goals = random.choices(score_range, weights=[1, higher_prob, higher_prob*2, higher_prob*3, higher_prob*4], k=1)[0]
        team2_goals = random.choices(score_range, weights=[1, lower_prob, lower_prob*2, lower_prob*3, lower_prob*4], k=1)[0]

        events = []
        # Half-Time event
        events.append({
            'minute': 45,
            'event': 'Half-Time',
            'team': '', 'player': '',
            'photo': '',
            'assist_text': '',
            'assist_photo': None
        })

        # Initialize clean sheets count
        clean_sheets = {team1_id: 0, team2_id: 0}

        # Simulate goals
        goal_minutes_team1 = random.sample(range(1, 90), team1_goals) if team1_goals > 0 else []
        goal_minutes_team2 = random.sample(range(1, 90), team2_goals) if team2_goals > 0 else []
        all_goal_minutes = sorted(goal_minutes_team1 + goal_minutes_team2)

        for minute in all_goal_minutes:
            if minute == 45:
                minute = 44 if 44 not in all_goal_minutes else 46

            scoring_team = team1_id if minute in goal_minutes_team1 else team2_id
            scorer = weighted_random_player(scoring_team, role='scorer')
            if scorer is None:
                scorer = weighted_random_player(scoring_team, role='')
            assist_text = ""
            assist_photo = None
            if scorer is not None and random.random() < 0.5:
                assister = weighted_random_player(scoring_team, role='assister')
                if assister is not None and assister['id_player'] != scorer['id_player']:
                    assist_text = f" assisted by {assister['player_name']}"
                    assist_photo = assister['player_image']

            event_entry = {
                'minute': minute,
                'event': 'Goal',
                'team': scorer['team'] if scorer is not None else '',
                'player': scorer['player_name'] if scorer is not None else '',
                'photo': scorer['player_image'] if scorer is not None else '',
                'assist_text': assist_text,
                'assist_photo': assist_photo
            }
            events.append(event_entry)

        # Check for clean sheets
        if team1_goals == 0:
            clean_sheets[team1_id] += 1
        if team2_goals == 0:
            clean_sheets[team2_id] += 1

        # Other events
        other_events_count = random.randint(5, 10)
        possible_events = ['Yellow Card', 'Red Card', 'Offside', 'Substitution', 'Save']
        event_weights = [0.25, 0.05, 0.25, 0.25, 0.2]

        for _ in range(other_events_count):
            chosen_event = random.choices(possible_events, weights=event_weights, k=1)[0]
            if chosen_event == 'Substitution':
                minute = random.randint(46, 89)
            else:
                minute = random.randint(1, 89)
                if minute == 45:
                    minute = 44

            chosen_team = random.choices([team1_id, team2_id], weights=[team1_prob, team2_prob], k=1)[0]
            if chosen_event == 'Substitution':
                sub_event = substitution_event(chosen_team, minute)
                if sub_event:
                    events.append(sub_event)
                else:
                    # If substitution fails, treat it as an offside
                    offside_player = weighted_random_player(chosen_team, 'offside')
                    if offside_player is None:
                        offside_player = weighted_random_player(chosen_team, '')
                    if offside_player is not None:
                        events.append({
                            'minute': minute,
                            'event': 'Offside',
                            'team': offside_player['team'] if offside_player is not None else '',
                            'player': offside_player['player_name'] if offside_player is not None else '',
                            'photo': offside_player['player_image'] if offside_player is not None else '',
                            'assist_text': "",
                            'assist_photo': None
                        })

            elif chosen_event in ['Yellow Card', 'Red Card']:
                carded_player = weighted_random_player(chosen_team, 'card')
                if carded_player is None:
                    carded_player = weighted_random_player(chosen_team, '')
                if carded_player is not None:
                    events.append({
                        'minute': minute,
                        'event': chosen_event,
                        'team': carded_player['team'],
                        'player': carded_player['player_name'],
                        'photo': carded_player['player_image'],
                        'assist_text': "",
                        'assist_photo': None
                    })
            elif chosen_event == 'Offside':
                offside_player = weighted_random_player(chosen_team, 'offside')
                if offside_player is None:
                    offside_player = weighted_random_player(chosen_team, '')
                if offside_player is not None:
                    events.append({
                        'minute': minute,
                        'event': 'Offside',
                        'team': offside_player['team'] if offside_player is not None else '',
                        'player': offside_player['player_name'] if offside_player is not None else '',
                        'photo': offside_player['player_image'] if offside_player is not None else '',
                        'assist_text': "",
                        'assist_photo': None
                    })
            elif chosen_event == 'Save':
                save_player = weighted_random_player(chosen_team, 'save')
                if save_player is not None and is_goalkeeper(save_player):
                    events.append({
                        'minute': minute,
                        'event': 'Save',
                        'team': save_player['team'],
                        'player': save_player['player_name'],
                        'photo': save_player['player_image'],
                        'assist_text': "",
                        'assist_photo': None
                    })

        # Sort events by minute
        events.sort(key=lambda x: x['minute'])

        team1 = teams_data[teams_data['team_id'] == team1_id]
        team2 = teams_data[teams_data['team_id'] == team2_id]

        if team1.empty:
            logger.error(f"Team ID {team1_id} not found in teams_data.csv.")
            return f"Error: Team ID {team1_id} not found in teams_data.csv."

        if team2.empty:
            logger.error(f"Team ID {team2_id} not found in teams_data.csv.")
            return f"Error: Team ID {team2_id} not found in teams_data.csv."

        team1 = team1.iloc[0]
        team2 = team2.iloc[0]

        return render_template(
            'result.html',
            team1=team1,
            team2=team2,
            team1_goals=team1_goals,
            team2_goals=team2_goals,
            events=events,
            team1_prob=f"{team1_prob*100:.2f}%",
            team2_prob=f"{team2_prob*100:.2f}%"
        )
    except Exception as e:
        logger.error(f"Error in /predict route: {e}")
        return f"Error: {e}"

# Helper functions for Group Stage and Playoffs
def simulate_match(team1_id, team2_id):
    if team1_id not in teams_data['team_id'].values or team2_id not in teams_data['team_id'].values:
        logger.warning(f"simulate_match received invalid team IDs: {team1_id}, {team2_id}")
        return None, None, 0, 0

    team1_stats_row = X[X['team_id'] == team1_id]
    team2_stats_row = X[X['team_id'] == team2_id]

    if team1_stats_row.empty or team2_stats_row.empty:
        logger.warning(f"simulate_match could not find stats for team IDs: {team1_id}, {team2_id}")
        return None, None, 0, 0

    team1_stats = team1_stats_row.iloc[0].mean()
    team2_stats = team2_stats_row.iloc[0].mean()

    total_stats = team1_stats + team2_stats
    team1_prob = team1_stats / total_stats if total_stats != 0 else 0.5
    team2_prob = team2_stats / total_stats if total_stats != 0 else 0.5

    # Adjust goal probabilities to favor higher-ranked team
    if team1_prob >= team2_prob:
        higher_prob = team1_prob
        lower_prob = team2_prob
    else:
        higher_prob = team2_prob
        lower_prob = team1_prob

    score_range = range(0, 5)
    team1_goals = random.choices(score_range, weights=[1, higher_prob, higher_prob*2, higher_prob*3, higher_prob*4], k=1)[0]
    team2_goals = random.choices(score_range, weights=[1, lower_prob, lower_prob*2, lower_prob*3, lower_prob*4], k=1)[0]

    if team1_goals > team2_goals:
        return team1_id, team2_id, team1_goals, team2_goals
    elif team2_goals > team1_goals:
        return team2_id, team1_id, team2_goals, team1_goals
    else:
        return None, None, team1_goals, team2_goals  # Draw

def rank_teams(standings):
    # Sort by points descending, then goal difference descending, then goals_for descending
    standings = standings.sort_values(by=['points', 'goal_difference', 'goals_for'], ascending=[False, False, False]).reset_index(drop=True)
    # Assign unique ranks
    standings['rank'] = standings.index + 1
    return standings

def simulate_playoff_match(team1_id, team2_id):
    winner_id, loser_id, team1_goals, team2_goals = simulate_match(team1_id, team2_id)
    if winner_id is not None:
        return winner_id
    else:
        # In case of a draw in knockout, decide randomly
        winner = random.choice([team1_id, team2_id])
        logger.debug(f"Match between {team1_id} and {team2_id} ended in a draw. Randomly selecting winner: {winner}")
        return winner

def simulate_playoff_round(teams):
    """
    Simulate a round of playoffs. Pair teams based on seedings.
    Higher seed vs Lower seed.
    """
    if len(teams) % 2 != 0:
        logger.warning("Odd number of teams in playoff round. One team will receive a bye.")
        # Assign a bye to the highest seeded team
        bye_team = teams.pop(0)
        logger.debug(f"Team {bye_team} receives a bye.")
        return [bye_team], []

    winners = []
    matchups = []
    # Sort teams based on initial standings (higher stats first)
    sorted_teams = X[X['team_id'].isin(teams)].sort_values(by=features, ascending=False)['team_id'].tolist()
    while len(sorted_teams) >= 2:
        team1 = sorted_teams.pop(0)  # Highest seed
        team2 = sorted_teams.pop(-1)  # Lowest seed
        winner = simulate_playoff_match(team1, team2)
        winners.append(winner)
        matchups.append({'team1': team1, 'team2': team2, 'winner': winner})
        logger.debug(f"Playoff Match: {team1} vs {team2} => Winner: {winner}")
    return winners, matchups

@app.route('/groups')
def groups():
    try:
        # Initialize standings
        standings = pd.DataFrame({
            'team_id': teams_data['team_id'],
            'team': teams_data['team'],
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0,
            'points': 0
        })

        # Initialize clean sheets count
        clean_sheets = {team_id: 0 for team_id in teams_data['team_id']}

        # Create all possible unique matches
        team_ids = teams_data['team_id'].tolist()
        all_matches = []
        for i in range(len(team_ids)):
            for j in range(i+1, len(team_ids)):
                all_matches.append((team_ids[i], team_ids[j]))

        # Shuffle matches to randomize
        random.shuffle(all_matches)

        # Assign matches ensuring each team plays exactly 8 matches
        matches_per_team = {team_id: 0 for team_id in team_ids}
        scheduled_matches = []

        for match in all_matches:
            team1, team2 = match
            if matches_per_team[team1] < 8 and matches_per_team[team2] < 8:
                scheduled_matches.append(match)
                matches_per_team[team1] += 1
                matches_per_team[team2] += 1
            # Stop scheduling if all teams have reached 8 matches
            if all(count == 8 for count in matches_per_team.values()):
                break

        logger.debug(f"Scheduled {len(scheduled_matches)} matches.")

        # Simulate each match
        for match in scheduled_matches:
            team1_id, team2_id = match
            winner_id, loser_id, team1_goals, team2_goals = simulate_match(team1_id, team2_id)

            # Update match statistics
            standings.loc[standings['team_id'] == team1_id, 'played'] += 1
            standings.loc[standings['team_id'] == team2_id, 'played'] += 1
            standings.loc[standings['team_id'] == team1_id, 'goals_for'] += team1_goals
            standings.loc[standings['team_id'] == team1_id, 'goals_against'] += team2_goals
            standings.loc[standings['team_id'] == team2_id, 'goals_for'] += team2_goals
            standings.loc[standings['team_id'] == team2_id, 'goals_against'] += team1_goals
            standings.loc[standings['team_id'] == team1_id, 'goal_difference'] = standings.loc[standings['team_id'] == team1_id, 'goals_for'].values[0] - standings.loc[standings['team_id'] == team1_id, 'goals_against'].values[0]
            standings.loc[standings['team_id'] == team2_id, 'goal_difference'] = standings.loc[standings['team_id'] == team2_id, 'goals_for'].values[0] - standings.loc[standings['team_id'] == team2_id, 'goals_against'].values[0]

            if winner_id is not None:
                standings.loc[standings['team_id'] == winner_id, 'wins'] += 1
                standings.loc[standings['team_id'] == winner_id, 'points'] += 3
                standings.loc[standings['team_id'] == loser_id, 'losses'] += 1
            else:
                standings.loc[standings['team_id'] == team1_id, 'draws'] += 1
                standings.loc[standings['team_id'] == team2_id, 'draws'] += 1
                standings.loc[standings['team_id'] == team1_id, 'points'] += 1
                standings.loc[standings['team_id'] == team2_id, 'points'] += 1

            # Update clean sheets
            if team2_goals == 0:
                clean_sheets[team1_id] += 1
            if team1_goals == 0:
                clean_sheets[team2_id] += 1

        # Rank teams uniquely
        standings = rank_teams(standings)

        # Merge standings with team logos
        standings = standings.merge(teams_data[['team_id', 'logo']], on='team_id', how='left')
        standings_display = standings.sort_values(by=['rank']).to_dict(orient='records')

        # Select top 24 teams
        top_24 = standings.head(24).copy()

        # Split top 8 and teams ranked 9-24
        top_8 = top_24.head(8).copy()
        knockout_teams = top_24.tail(16).copy()

        logger.debug("Top 8 teams:")
        logger.debug(top_8[['team_id', 'team', 'points', 'rank']])

        logger.debug("Knockout Playoff teams (9-24):")
        logger.debug(knockout_teams[['team_id', 'team', 'points', 'rank']])

        # Knockout Playoffs (Teams ranked 9-24)
        knockout_playoff_teams = knockout_teams['team_id'].tolist()
        logger.debug(f"Knockout Playoff teams: {knockout_playoff_teams}")
        playoff_winners, playoff_matchups = simulate_playoff_round(knockout_playoff_teams)

        logger.debug(f"Playoff winners: {playoff_winners}")

        # Round of 16: Top 8 vs Playoff Winners
        # Sort playoff winners based on their initial standings (higher points first)
        playoff_winners_stats = standings[standings['team_id'].isin(playoff_winners)].sort_values(by=['points', 'goal_difference', 'goals_for'], ascending=False)['team_id'].tolist()
        round_of_16_matchups = []
        sorted_top_8 = top_8.sort_values(by=['rank']).team_id.tolist()
        sorted_playoff_winners = playoff_winners_stats.copy()

        for i in range(len(sorted_top_8)):
            if not sorted_playoff_winners:
                logger.warning("Not enough playoff winners to pair with top 8 teams.")
                break
            team1 = sorted_top_8[i]  # Top seed
            team2 = sorted_playoff_winners.pop(-1)  # Lowest playoff winner
            winner = simulate_playoff_match(team1, team2)
            round_of_16_matchups.append({'team1': team1, 'team2': team2, 'winner': winner})
            logger.debug(f"Round of 16 Match: {team1} vs {team2} => Winner: {winner}")

        # Quarterfinals
        quarterfinal_winners = []
        quarterfinal_matchups = []
        round_of_16_winners = [match['winner'] for match in round_of_16_matchups]
        random.shuffle(round_of_16_winners)
        for i in range(0, len(round_of_16_winners), 2):
            if i+1 < len(round_of_16_winners):
                team1 = round_of_16_winners[i]
                team2 = round_of_16_winners[i+1]
                winner = simulate_playoff_match(team1, team2)
                quarterfinal_winners.append(winner)
                quarterfinal_matchups.append({'team1': team1, 'team2': team2, 'winner': winner})
                logger.debug(f"Quarterfinal Match: {team1} vs {team2} => Winner: {winner}")

        # Semifinals
        semifinal_winners = []
        semifinal_matchups = []
        semifinal_teams = quarterfinal_winners.copy()
        random.shuffle(semifinal_teams)
        for i in range(0, len(semifinal_teams), 2):
            if i+1 < len(semifinal_teams):
                team1 = semifinal_teams[i]
                team2 = semifinal_teams[i+1]
                winner = simulate_playoff_match(team1, team2)
                semifinal_winners.append(winner)
                semifinal_matchups.append({'team1': team1, 'team2': team2, 'winner': winner})
                logger.debug(f"Semifinal Match: {team1} vs {team2} => Winner: {winner}")

        # Finals
        final_winners = []
        final_matchups = []
        if len(semifinal_winners) == 2:
            team1, team2 = semifinal_winners
            winner = simulate_playoff_match(team1, team2)
            final_winners.append(winner)
            final_matchups.append({'team1': team1, 'team2': team2, 'winner': winner})
            logger.debug(f"Final Match: {team1} vs {team2} => Winner: {winner}")

        # Determine Champion
        if final_winners:
            champion_id = final_winners[0]
            champion_team_row = teams_data[teams_data['team_id'] == champion_id]
            if champion_team_row.empty:
                logger.error(f"Champion team ID {champion_id} not found in teams_data.csv.")
                champion_team = "Unknown Champion"
                champion_logo = ""
            else:
                champion_team = champion_team_row.iloc[0]['team']
                champion_logo = champion_team_row.iloc[0]['logo']
        else:
            champion_team = "No Champion Determined"
            champion_logo = ""

        # Prepare playoff matchups for display with logos
        def prepare_match_display(matchups):
            display = []
            for m in matchups:
                team1 = teams_data[teams_data['team_id'] == m['team1']]
                team2 = teams_data[teams_data['team_id'] == m['team2']]
                winner = teams_data[teams_data['team_id'] == m['winner']]
                if team1.empty or team2.empty or winner.empty:
                    logger.warning(f"Invalid matchup data: {m}")
                    match_str = "Invalid Match"
                    winner_str = "Unknown"
                    team1_logo = ""
                    team2_logo = ""
                else:
                    match_str = f"{team1.iloc[0]['team']} vs {team2.iloc[0]['team']}"
                    winner_str = winner.iloc[0]['team']
                    team1_logo = team1.iloc[0]['logo']
                    team2_logo = team2.iloc[0]['logo']
                display.append({'match': match_str, 'winner': winner_str, 'team1_logo': team1_logo, 'team2_logo': team2_logo})
            return display

        playoff_display = prepare_match_display(playoff_matchups)
        round_of_16_display = prepare_match_display(round_of_16_matchups)
        quarterfinal_display = prepare_match_display(quarterfinal_matchups)
        semifinal_display = prepare_match_display(semifinal_matchups)
        final_display = prepare_match_display(final_matchups)

        # Compute Top 10 Goalscorers (Randomized)x
        top_goalscorers = goals_data.groupby('id_player')['goals'].sum().reset_index()
        top_goalscorers['random_weight'] = top_goalscorers['goals'] * np.random.uniform(1.0, 2.0, len(top_goalscorers))
        top_goalscorers = top_goalscorers.sort_values(by='random_weight', ascending=False).head(10)
        top_goalscorers = top_goalscorers.merge(players_data, on='id_player', how='left').merge(teams_data, left_on='id_team', right_on='team_id', how='left')
        top_goalscorers = top_goalscorers[['player_name', 'goals', 'player_image', 'team']].dropna().to_dict(orient='records')

        # Compute Top 10 Assisters (Randomized)
        top_assisters = attacking_data.groupby('id_player')['assists'].sum().reset_index()
        top_assisters['random_weight'] = top_assisters['assists'] * np.random.uniform(1.0, 2.0, len(top_assisters))
        top_assisters = top_assisters.sort_values(by='random_weight', ascending=False).head(10)
        top_assisters = top_assisters.merge(players_data, on='id_player', how='left').merge(teams_data, left_on='id_team', right_on='team_id', how='left')
        top_assisters = top_assisters[['player_name', 'assists', 'player_image', 'team']].dropna().to_dict(orient='records')


        # Compute Top 10 Clean Sheets (Goalkeepers)
        # Assign clean sheets based on simulated matches
        # Map clean sheets to goalkeepers
        goalkeeper_players = players_data[players_data['field_position'].str.lower().isin(['goalkeeper', 'gk', 'goalie'])]

        # Merge with teams to get team information
        goalkeeper_players = goalkeeper_players.merge(teams_data, left_on='id_team', right_on='team_id', how='left')

        # Assign clean sheets to goalkeepers
        goalkeeper_players['clean_sheets'] = goalkeeper_players['team_id'].map(clean_sheets).fillna(0).astype(int)

        # Select relevant columns
        top_clean_sheets = goalkeeper_players[['player_name', 'clean_sheets', 'player_image', 'team']]

        # Sort and take top 10
        top_clean_sheets = top_clean_sheets.sort_values(by='clean_sheets', ascending=False).head(10).to_dict(orient='records')

        # Debugging Step: Check if top_clean_sheets is empty
        if not top_clean_sheets:
            logger.warning("Top Clean Sheets table is empty. Possible reasons:")
            logger.warning("- No goalkeepers identified with the specified positions.")
            logger.warning("- 'clean_sheets' data might be missing or zero for all goalkeepers.")
            logger.warning("- Data merging issues might have occurred.")

        return render_template('group.html',
                               standings=standings_display,
                               playoff=playoff_display,
                               round_of_16=round_of_16_display,
                               quarterfinal=quarterfinal_display,
                               semifinal=semifinal_display,
                               final=final_display,
                               champion=champion_team,
                               champion_logo=champion_logo,
                               top_goalscorers=top_goalscorers,
                               top_assisters=top_assisters,
                               top_clean_sheets=top_clean_sheets)
    except Exception as e:
        logger.error(f"Error in /groups route: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
