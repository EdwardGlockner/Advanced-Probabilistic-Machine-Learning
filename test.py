import re
import pandas as pd
import csv  # Import the csv module

def format_pgn_to_csv(input_file, output_file):
    # Read PGN data from the input file
    with open(input_file, 'r') as file:
        pgn_data = file.read()

    # Regular expressions for extracting relevant data
    event_pattern = r'\[Event "(.*?)"\]'
    site_pattern = r'\[Site "(.*?)"\]'
    date_pattern = r'\[Date "(.*?)"\]'
    round_pattern = r'\[Round "(.*?)"\]'
    white_player_pattern = r'\[White "(.*?)"\]'
    black_player_pattern = r'\[Black "(.*?)"\]'
    white_elo_pattern = r'\[WhiteElo "(.*?)"\]'
    black_elo_pattern = r'\[BlackElo "(.*?)"\]'
    result_pattern = r'\[Result "(.*?)"\]'
    
    # Split the data into individual games based on the '[Event' line
    games = pgn_data.strip().split('\n\n')

    # Prepare a list to hold the formatted game results
    formatted_games = []

    for game in games:
        # Extract game details using regex
        event = re.search(event_pattern, game)
        site = re.search(site_pattern, game)
        date = re.search(date_pattern, game)
        round_number = re.search(round_pattern, game)
        white_player = re.search(white_player_pattern, game)
        black_player = re.search(black_player_pattern, game)
        white_elo = re.search(white_elo_pattern, game)
        black_elo = re.search(black_elo_pattern, game)
        result = re.search(result_pattern, game)

        if all([event, site, date, round_number, white_player, black_player, white_elo, black_elo, result]):
            # Extract player names and ELO ratings
            w_player = white_player.group(1)
            b_player = black_player.group(1)
            w_elo = white_elo.group(1)
            b_elo = black_elo.group(1)
            game_result = result.group(1)

            # Replace result format if necessary (e.g., 1-0, 0-1, 1/2-1/2 to desired format)
            formatted_result = game_result.replace('1-0', '1 – 0').replace('0-1', '0 – 1').replace('1/2-1/2', '½ – ½')

            # Format the result for CSV
            formatted_games.append(f"{w_player},{w_elo},{b_player},{b_elo},{formatted_result}")

    # Save the formatted results to a CSV file without quotes
    df = pd.DataFrame(formatted_games, columns=["Game"])
    df.to_csv(output_file, index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')

input_file = 'Stavanger2024.pgn'  # Specify your input PGN file name
output_file = 'chess_games.csv'    # Specify the desired output CSV file name

format_pgn_to_csv(input_file, output_file)

