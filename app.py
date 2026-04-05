import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("ipl_winner_model.pkl", "rb"))

st.title("🏏 IPL Match Winner Predictor")

teams = [
'Chennai Super Kings',
'Delhi Capitals',
'Gujarat Titans',
'Kolkata Knight Riders',
'Lucknow Super Giants',
'Mumbai Indians',
'Punjab Kings',
'Rajasthan Royals',
'Royal Challengers Bangalore',
'Sunrisers Hyderabad'
]

# User Inputs
batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)

venue = st.text_input("Venue")
city = st.text_input("City")

toss_winner = st.selectbox("Toss Winner", teams)
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])


if st.button("Predict Winner"):

    # Encode teams
    bt = teams.index(batting_team)
    bw = teams.index(bowling_team)
    tw = teams.index(toss_winner)

    # Encode toss decision
    td = 0 if toss_decision == "bat" else 1

    # Placeholder features (must match training feature count)
    vn = 0
    ct = 0

    batting_win_rate = 0.5
    bowling_win_rate = 0.5

    h2h_batting = 1
    h2h_bowling = 1

    batting_venue_rate = 0.5
    bowling_venue_rate = 0.5

    # Model input
    input_data = np.array([[
        bt,
        bw,
        vn,
        ct,
        tw,
        td,
        batting_win_rate,
        bowling_win_rate,
        h2h_batting,
        h2h_bowling,
        batting_venue_rate,
        bowling_venue_rate
    ]])

    # Prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Extract probabilities
    bt_prob = probabilities[0][bt]
    bw_prob = probabilities[0][bw]

    total = bt_prob + bw_prob

    bt_prob = (bt_prob / total) * 100
    bw_prob = (bw_prob / total) * 100

    predicted_team = teams[prediction[0]]

    # Ensure winner is one of the playing teams
    if predicted_team not in [batting_team, bowling_team]:
        if bt_prob > bw_prob:
            predicted_team = batting_team
        else:
            predicted_team = bowling_team

    # Display Results
    st.success("Predicted Winner: " + predicted_team)

    st.write(f"{batting_team} Win Probability: {bt_prob:.2f}%")
    st.write(f"{bowling_team} Win Probability: {bw_prob:.2f}%")