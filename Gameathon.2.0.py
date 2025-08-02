import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

                                                                                                         #BOWLING
# Define file path
file_path = "C:/STORAGE/Code/IPL/New/deliveries.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv(file_path)
columns_needed = ["match_id", "bowler", "batter", "batsman_runs", "is_wicket", "dismissal_kind"]
df = df[columns_needed].fillna(0)

# Split 2024 data from previous years
df_2024 = df.iloc[243819:].copy()
df_prev = df.iloc[:].copy()

# Function to calculate bowling stats per match
def calculate_bowling_stats(data):
    data["dot_balls"] = (data["batsman_runs"] == 0).astype(int)
    data["bowled_lbw_wickets"] = data["dismissal_kind"].isin(["bowled", "lbw"]).astype(int)
    
    bowling_stats = data.groupby(["match_id", "bowler"]).agg(
        total_wickets=("is_wicket", "sum"),
        bowled_lbw_wickets=("bowled_lbw_wickets", "sum"),
        dot_balls=("dot_balls", "sum")
    ).reset_index()     
    
    bowling_stats["bonus_3_wickets"] = (bowling_stats["total_wickets"] >= 3).astype(int) * 4
    bowling_stats["bonus_4_wickets"] = (bowling_stats["total_wickets"] >= 4).astype(int) * 8
    bowling_stats["bonus_5_wickets"] = (bowling_stats["total_wickets"] >= 5).astype(int) * 12
    
    bowling_stats["bowling_points"] = (
        bowling_stats["total_wickets"] * 30 +
        bowling_stats["bowled_lbw_wickets"] * 8 +
        bowling_stats["dot_balls"] * 1 +
        bowling_stats["bonus_3_wickets"] +
        bowling_stats["bonus_4_wickets"] +
        bowling_stats["bonus_5_wickets"]
    )
    
    return bowling_stats

bowl_stats = calculate_bowling_stats(df_prev)
bowl_stats_2024 = calculate_bowling_stats(df_2024)

# Rename columns for consistency
bowl_stats = bowl_stats.rename(columns={"bowler": "Player"})
bowl_stats_2024 = bowl_stats_2024.rename(columns={"bowler": "Player"})


# Define actual average bowling points for target players
actual_bowling_points = {
    "Abhinandan Singh": 0,
    "B Kumar": 32.39,
    "D Padikkal": 0,
    "N Thushara": 35.43,
    "Jacob Bethell": 0,
    "JM Sharma": 0,
    "JR Hazlewood": 40.22,
    "KH Pandya": 18.27,
    "LS Livingstone": 8.56,
    "L Ngidi": 57,
    "Manoj Bhandage": 0,
    "Mohit Rathee": 0,
    "PD Salt": 0,
    "RM Patidar": 0,
    "Rasikh Dar": 0,
    "R Shepherd": 12,
    "Suyash Sharma": 23.38,
    "Swapnil Singh": 15,
    "Swastik Chhikara": 0,
    "TH David": 0,
    "V Kohli": 0.48,
    "Yash Dayal": 30.29,

    "Andre Siddarth": 0,
    "A Kamboj": 20,
    "DJ Hooda": 2.54,
    "DP Conway": 0,
    "Gurjapneeth Singh": 0,
    "J Overton": 0,
    "KL Nagarkoti": 12.5,
    "H Ahmed": 40.35,
    "MS Dhoni": 0,
    "M Pathirana": 52.2,
    "Mukesh Choudhary": 35.71,
    "NT Ellis": 34.75,
    "Noor Ahmad": 31.48,
    "R Ravindra": 0,
    "RA Tripathi": 0,
    "Ramkrishna Ghosh": 0,
    "R Ashwin": 26.02,
    "RA Jadeja": 20.57,
    "RD Gaikwad": 0,
    "SM Curran": 30.1,
    "Shaik Rasheed": 0,
    "S Dube": 2.31,
    "S Gopal": 30.77,
    "Vansh Bedi": 0,
    "V Shankar": 3.75,

    "Abishek Porel": 0,
    "AR Sharma": 0,
    "AR Patel": 24.92,
    "DG Nalkande": 30,
    "D Ferreira": 0,
    "F du Plessis": 0,
    "Ajay Mandal": 0,
    "J Fraser-McGurk": 0,
    "KL Rahul": 0,
    "KK Nair": 0,
    "Kuldeep Yadav": 31.98,
    "Madhav Tiwari": 0,
    "Manvanth Kumar": 0,
    "MA Starc": 38.68,
    "MM Sharma": 36.32,
    "Mukesh Kumar": 36.6,
    "PVD Chameera": 20.77,
    "Sameer Rizvi": 0,
    "T Natarajan": 33.74,
    "Tripurana Vijay": 0,
    "T Stubbs": 6.67,
    "Vipraj Nigam": 0,

    "Shubman Gill": 0,
    "JC Buttler": 0,
    "K Kushagra": 0,
    "Anuj Rawat": 0,
    "SE Rutherford": 3,
    "GD Phillips": 7.5,
    "MK Lomror": 0.75,
    "Washington Sundar": 18.9,
    "R Sai Kishore": 40.2,
    "J Yadav": 12,
    "Karim Janat": 0,
    "B Sai Sudharsan": 0,
    "M Shahrukh Khan": 0,
    "K Rabada": 45.33,
    "Mohammed Siraj": 30.99,
    "M Prasidh Krishna": 30.08,
    "MJ Suthar": 0,
    "N Sindhu": 0,
    "Arshad Khan": 17.43,
    "G Coetzee": 40.6,
    "Gurnoor Brar": 0,
    "I Sharma": 25.65,
    "K Khejroliya": 22.5,
    "R Tewatia": 9.86,
    "Rashid Khan": 37.15,

    "A Manohar": 0,
    "Abhishek Sharma": 5.24,
    "Aniket Verma": 0,
    "A Taide": 0,
    "Eshan Malinga": 0,
    "HV Patel": 39.34,
    "H Klaasen": 0,
    "Ishan Kishan": 0,
    "JD Unadkat": 29.31,
    "Kamindu Mendis": 0,
    "Mohammed Shami": 35.58,
    "K Nitish Kumar Reddy": 6,
    "PJ Cummins": 33.69,
    "RD Chahar": 29.41,
    "Sachin Baby": 3.16,
    "Simarjeet Singh": 27.4,
    "TM Head": 2.4,
    "Wiaan Mulder": 0,
    "Zeeshan Ansari": 0,
    "Ravichandran Smaran": 0,

    "Aaron Hardie": 0,
    "Arshdeep Singh": 36.31,
    "Azmatullah Omarzai": 17.14,
    "GJ Maxwell": 8.37,
    "Harnoor Pannu": 0,
    "Harpreet Brar": 19.07,
    "KR Sen": 36.33,
    "LH Ferguson": 31.47,
    "M Jansen": 29.33,
    "MP Stoinis": 13.69,
    "Musheer Khan": 0,
    "JP Inglis": 0,
    "N Wadhera": 0,
    "Prabhsimran Singh": 0,
    "P Dubey": 7.5,
    "Priyansh Arya": 0,
    "Pyla Avinash": 0,
    "Shashank Singh": 1.25,
    "SS Iyer": 0,
    "Suryansh Shedge": 0,
    "V Vyshak": 35.82,
    "Vishnu Vinod": 0,
    "Xavier Bartlett": 0,
    "Yash Thakur": 40.63,
    "YS Chahal": 39.66,

    "RG Sharma": 1.81,
    "SA Yadav": 0,
    "Robin Minz": 0,
    "Tilak Varma": 0,
    "HH Pandya": 14.16,
    "WG Jacks": 7.5,
    "MJ Santner": 25,
    "RA Bawa": 0,
    "TA Boult": 36.6,
    "KV Sharma": 27.67,
    "DL Chahar": 29.95,
    "RJW Topley": 30,
    "AS Tendulkar": 18,
    "Mujeeb Ur Rahman": 30.21,
    "JJ Bumrah": 38.96,
    "Ryan Rickelton": 0,
    "Shrijith Krishnan": 0,
    "Bevon Jacobs": 0,
    "Naman Dhir": 0,
    "Vignesh Puthur": 0,
    "Corbin Bosch": 0,
    "Ashwani Kumar": 0,
    "V Satyanarayan Penmetsa": 0,

     "RR Pant": 0,
    "DA Miller": 0,
    "AK Markram": 1.36,
    "N Pooran": 0,
    "MR Marsh": 27.38,
    "Abdul Samad": 1.2,
    "RS Hangargekar": 47,
    "AA Kulkarni": 0,
    "A Badoni": 1.43,
    "Avesh Khan": 36.19,
    "Akash Deep": 28.25,
    "Akash Singh": 21.43,
    "S Joseph": 0,
    "MP Yadav": 54.5,
    "MP Breetzke": 0,
    "Himmat Singh": 0,
    "A Juyal": 0,
    "Shahbaz Ahmed": 11.39,
    "Y Chaudhary": 0,
    "SN Thakur": 31.71,
    "M Siddharth": 18,
    "DS Rathi": 0,
    "Prince Yadav": 0,
    "Ravi Bishnoi": 29.56,

    "A Madhwal": 47.23,
    "Ashok Sharma": 0,
    "DC Jurel": 0,
    "Fazalhaq Farooqi": 25.71,
    "JC Archer": 37.2,
    "K Kartikeya": 25,
    "Kunal Rathode": 0,
    "KT Maphaka": 15,
    "M Theekshana": 28.81,
    "N Rana": 2.8,
    "PW Hasaranga": 42.23,
    "R Parag": 1.71,
    "Sandeep Sharma": 33.78,
    "SV Samson": 0,
    "SO Hetmyer": 0,
    "SB Dubey": 0,
    "TU Deshpande": 35.67,
    "V Suryavanshi": 0,
    "YBK Jaiswal": 0,
    "Yudhvir Singh": 24,

    "AM Rahane": 0.16,
    "AD Russell": 27.86,
    "A Raghuvanshi": 0,
    "A Nortje": 39.74,
    "AS Roy": 13.64,
    "C Sakariya": 0,
    "Harshit Rana": 36.67,
    "LS Sisodia": 0,
    "MK Pandey": 0,
    "M Markande": 30.76,
    "MM Ali": 15.97,
    "Q de Kock": 0,
    "Rahmanullah Gurbaz": 0,
    "Ramandeep Singh": 9.2,
    "RK Singh": 0,
    "R Powell": 1.11,
    "SH Johnson": 24,
    "SP Narine": 31.48,
    "Umran Malik": 35.62,
    "VG Arora": 28.7,
    "CV Varun": 36.08,
    "VR Iyer": 1.76
}

# Merge actual average points
bowl_stats["Actual aAvg Points"] = bowl_stats["Player"].map(actual_bowling_points)
bowl_stats_2024["Actual aAvg Points"] = bowl_stats_2024["Player"].map(actual_bowling_points)

# Remove missing values
bowl_stats = bowl_stats.dropna()
bowl_stats_2024 = bowl_stats_2024.dropna()

# Filter 2024 players in overall data
aplayers_2024 = set(bowl_stats_2024["Player"])
bowl_stats = bowl_stats[bowl_stats["Player"].isin(aplayers_2024)]

# Define features and target
afeatures = ["dot_balls", "bowled_lbw_wickets", "total_wickets"]
atarget = "Actual aAvg Points"

# Apply weight to target variable
bowl_stats["aWeight"] = 0.7  # 70% weight to overall performance
bowl_stats_2024["aWeight"] = 1.3  # 130% weight to recent 2024 performance

bowl_stats_combined = pd.concat([bowl_stats, bowl_stats_2024], ignore_index=True)
bowl_stats_combined[atarget] = bowl_stats_combined[atarget] * bowl_stats_combined["aWeight"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    bowl_stats_combined[afeatures], bowl_stats_combined[atarget], test_size=0.2, random_state=77
)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, learning_rate=0.05, max_depth=7)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
# print("Test MSE:", mean_squared_error(y_test, y_pred))

# Predict next match stats
alatest_stats = bowl_stats_combined.groupby("Player").last().reset_index()
X_pred = alatest_stats[afeatures]
alatest_stats["Estimated aPoints Next Match"] = model.predict(X_pred)

# Apply correction factor
alatest_stats["Estimated aPoints Next Match"] = alatest_stats.apply(
    lambda row: (row["Estimated aPoints Next Match"] + row["Actual aAvg Points"]) / 2, axis=1
)

# Assign 0 points to players without valid batting data
alatest_stats["Estimated aPoints Next Match"] = alatest_stats["Estimated aPoints Next Match"].fillna(0)

atop_players = alatest_stats.sort_values(by="Estimated aPoints Next Match", ascending=False)

# Create a complete bowling prediction dictionary with 0s for missing players
model_bowling_points = {}

for player in actual_bowling_points.keys():
    points = alatest_stats.loc[alatest_stats["Player"] == player, "Estimated aPoints Next Match"]
    model_bowling_points[player] = round(points.values[0], 2) if not points.empty else 0.0
                                                                                                            #BATTING

from sklearn.metrics import mean_squared_error

# Ensure necessary columns exist
#columns_needed = ["match_id", "batter", "batsman_runs"]
#df = df[columns_needed].fillna(0)

# Split 2024 data from previous years
df_2024 = df.iloc[243819:].copy()
df_prev = df.iloc[:].copy()

# Calculate player stats per match
def calculate_batting_stats(data):
    data["num_fours"] = (data["batsman_runs"] == 4).astype(int)
    data["num_sixes"] = (data["batsman_runs"] == 6).astype(int)
    return data.groupby(["match_id", "batter"]).agg(
        total_runs=("batsman_runs", "sum"),
        num_fours=("num_fours", "sum"),
        num_sixes=("num_sixes", "sum")
    ).reset_index()

bat_stats = calculate_batting_stats(df_prev)
bat_stats_2024 = calculate_batting_stats(df_2024)

# Rename columns for consistency
bat_stats = bat_stats.rename(columns={"batter": "Player"})
bat_stats_2024 = bat_stats_2024.rename(columns={"batter": "Player"})

# Define actual average points
actual_avg_points = {
    "Abhinandan Singh": 0,
    "B Kumar": 2.52,
    "D Padikkal": 39.98,
    "N Thushara": 0,
    "Jacob Bethell": 0,
    "JM Sharma": 30.3,
    "JR Hazlewood": 0.85,
    "KH Pandya": 20.45,
    "LS Livingstone": 42.23,
    "L Ngidi": 0,
    "Manoj Bhandage": 0,
    "Mohit Rathee": 1,
    "PD Salt": 57.19,
    "RM Patidar": 51.67,
    "Rasikh Dar": 0,
    "R Shepherd": 20.3,
    "Suyash Sharma": 0,
    "Swapnil Singh": 5.79,
    "Swastik Chhikara": 0,
    "TH David": 28.61,
    "V Kohli": 51.56,
    "Yash Dayal": 0,

    "Andre Siddarth": 0,
    "A Kamboj": 0.67,
    "DJ Hooda": 19.31,
    "DP Conway": 68.35,
    "Gurjapneeth Singh": 0,
    "J Overton": 0,
    "KL Nagarkoti": 2.17,
    "KK Ahmed": 0.02,
    "MS Dhoni": 31.81,
    "M Pathirana": 0,
    "Mukesh Choudhary": 0.71,
    "NT Ellis": 1.56,
    "Noor Ahmad": 0.91,
    "R Ravindra": 39,
    "RA Tripathi": 39.24,
    "Ramkrishna Ghosh": 0,
    "R Ashwin": 5.75,
    "RA Jadeja": 18.69,
    "RD Gaikwad": 60.03,
    "SM Curran": 24.42,
    "Shaik Rasheed": 0,
    "S Dube": 38.74,
    "S Gopal": 5.15,
    "Vansh Bedi": 0,
    "V Shankar": 24.35,

     "Abishek Porel": 34.11,
    "AR Sharma": 29.73,
    "AR Patel": 17.19,
    "DG Nalkande": 3,
    "D Ferreira": 4,
    "F du Plessis": 52.05,
    "Ajay Mandal": 0,
    "J Fraser-McGurk": 73.11,
    "KL Rahul": 58.7,
    "KK Nair": 32.29,
    "Kuldeep Yadav": 3.15,
    "Madhav Tiwari": 0,
    "Manvanth Kumar": 0,
    "MA Starc": 3.63,
    "MM Sharma": 1.61,
    "Mukesh Kumar": 0.5,
    "PVD Chameera": 5.62,
    "Sameer Rizvi": 9.88,
    "T Natarajan": 0.05,
    "Tripurana Vijay": 0,
    "T Stubbs": 38.28,
    "Vipraj Nigam": 0,

    "Shubman Gill": 50.82,
    "JC Buttler": 57.98,
    "K Kushagra": 0.75,
    "Anuj Rawat": 21.42,
    "SE Rutherford": 16.6,
    "GD Phillips": 14.13,
    "MK Lomror": 21.18,
    "Washington Sundar": 9.33,
    "R Sai Kishore": 2.5,
    "J Yadav": 2.7,
    "Karim Janat": 0,
    "B Sai Sudharsan": 65.92,
    "M Shahrukh Khan": 23.08,
    "K Rabada": 3.98,
    "Mohammed Siraj": 1.86,
    "M Prasidh Krishna": 0.18,
    "MJ Suthar": 1,
    "N Sindhu": 0,
    "Arshad Khan": 12.46,
    "G Coetzee": 0,
    "Gurnoor Brar": 0,
    "I Sharma": 0,
    "K Khejroliya": 0,
    "R Tewatia": 17.3,
    "Rashid Khan": 7.71,

     "A Manohar": 19.74,
    "Abhishek Sharma": 37.83,
    "Aniket Verma": 0,
    "A Taide": 46.56,
    "Eshan Malinga": 0,
    "HV Patel": 3.8,
    "H Klaasen": 47.46,
    "Ishan Kishan": 42.91,
    "JD Unadkat": 2.85,
    "Kamindu Mendis": 0,
    "Mohammed Shami": 1,
    "K Nitish Kumar Reddy": 33.67,
    "PJ Cummins": 15.09,
    "RD Chahar": 2.71,
    "Sachin Baby": 11.47,
    "Simarjeet Singh": 0.7,
    "TM Head": 54.72,
    "Wiaan Mulder": 0,
    "Zeeshan Ansari": 0,
    "Ravichandran Smaran": 0,

    "Aaron Hardie": 0,
    "Arshdeep Singh": 0.69,
    "Azmatullah Omarzai": 8.86,
    "GJ Maxwell": 35.84,
    "Harnoor Pannu": 0,
    "Harpreet Brar": 8.85,
    "JP Inglis": 0,
    "KR Sen": 0,
    "LH Ferguson": 2.31,
    "M Jansen": 4.67,
    "MP Stoinis": 32.25,
    "Musheer Khan": 0,
    "N Wadhera": 30.1,
    "Prabhsimran Singh": 39.71,
    "P Dubey": 7.75,
    "Priyansh Arya": 0,
    "Pyla Avinash": 0,
    "Shashank Singh": 30.04,
    "SS Iyer": 43.49,
    "Suryansh Shedge": 0,
    "V Vyshak": 1.82,
    "Vishnu Vinod": 14.33,
    "Xavier Bartlett": 0,
    "Yash Thakur": 0,
    "YS Chahal": 0.23,

     "RG Sharma": 43.08,
    "SA Yadav": 40.87,
    "Robin Minz": 0,
    "Tilak Varma": 50.32,
    "HH Pandya": 30.49,
    "WG Jacks": 52.75,
    "MJ Santner": 5.78,
    "RA Bawa": 7.5,
    "TA Boult": 1.16,
    "KV Sharma": 6.35,
    "DL Chahar": 1.53,
    "RJW Topley": 0.6,
    "AS Tendulkar": 3.8,
    "Mujeeb Ur Rahman": 1.05,
    "JJ Bumrah": 0.71,
    "Ryan Rickelton": 0,
    "Shrijith Krishnan": 0,
    "Bevon Jacobs": 0,
    "Naman Dhir": 48.57,
    "Vignesh Puthur": 0,
    "Corbin Bosch": 0,
    "Ashwani Kumar": 0,
    "V Satyanarayan Penmetsa": 0,

    "RR Pant": 49.98,
    "DA Miller": 35.94,
    "AK Markram": 34.93,
    "N Pooran": 40.09,
    "MR Marsh": 26.43,
    "Abdul Samad": 18.82,
    "RS Hangargekar": 0,
    "AA Kulkarni": 8.5,
    "A Badoni": 23.67,
    "Avesh Khan": 1.13,
    "Akash Deep": 4.13,
    "Akash Singh": 0,
    "S Joseph": 0,
    "MP Yadav": 0,
    "MP Breetzke": 0,
    "Himmat Singh": 0,
    "A Juyal": 0,
    "Shahbaz Ahmed": 14.63,
    "Y Chaudhary": 0,
    "SN Thakur": 5.13,
    "M Siddharth": 0,
    "DS Rathi": 0,
    "Prince Yadav": 0,
    "Ravi Bishnoi": 0.56,

      "A Madhwal": 0.31,
    "Ashok Sharma": 0,
    "DC Jurel": 20.54,
    "Fazalhaq Farooqi": 0.43,
    "JC Archer": 8.18,
    "K Kartikeya": 1.08,
    "Kunal Rathode": 0,
    "KT Maphaka": 0,
    "M Theekshana": 0.48,
    "N Rana": 42.13,
    "PW Hasaranga": 4.08,
    "R Parag": 27.39,
    "Sandeep Sharma": 0.55,
    "SV Samson": 43.45,
    "SO Hetmyer": 28.76,
    "SB Dubey": 14.25,
    "TU Deshpande": 0.97,
    "V Suryavanshi": 0,
    "YBK Jaiswal": 54.32,
    "Yudhvir Singh": 7.6,

     "AM Rahane": 40.19,
    "AD Russell": 35.48,
    "A Raghuvanshi": 28.3,
    "A Nortje": 1.59,
    "AS Roy": 3.45,
    "C Sakariya": 0,
    "Harshit Rana": 0.1,
    "LS Sisodia": 0,
    "MK Pandey": 35.31,
    "M Markande": 1.78,
    "MM Ali": 29.73,
    "Q de Kock": 50.16,
    "Rahmanullah Gurbaz": 36.79,
    "Ramandeep Singh": 14,
    "RK Singh": 31.93,
    "R Powell": 22.96,
    "SH Johnson": 2,
    "SP Narine": 16.05,
    "Umran Malik": 1.5,
    "VG Arora": 0.45,
    "CV Varun": 0.46,
    "VR Iyer": 44.63
}

# Merge actual average points
bat_stats["Actual Avg Points"] = bat_stats["Player"].map(actual_avg_points)
bat_stats_2024["Actual Avg Points"] = bat_stats_2024["Player"].map(actual_avg_points)

# Remove missing values
bat_stats = bat_stats.dropna()
bat_stats_2024 = bat_stats_2024.dropna()

# Filter 2024 players in overall data
bplayers_2024 = set(bat_stats_2024["Player"])
bat_stats = bat_stats[bat_stats["Player"].isin(bplayers_2024)]

# Define features and target
features = ["total_runs", "num_fours", "num_sixes"]
target = "Actual Avg Points"

# Apply weight to target variable
bat_stats["Weight"] = 0.7  # 70% weight to overall performance
bat_stats_2024["Weight"] = 1.3  # 130% weight to recent 2024 performance

bat_stats_combined = pd.concat([bat_stats, bat_stats_2024], ignore_index=True)
bat_stats_combined[target] = bat_stats_combined[target] * bat_stats_combined["Weight"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    bat_stats_combined[features], bat_stats_combined[target], test_size=0.2, random_state=77
)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, learning_rate=0.05, max_depth=7)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
# print("Test MSE:", mean_squared_error(y_test, y_pred))

# Predict next match stats
latest_stats = bat_stats_combined.groupby("Player").last().reset_index()
X_pred = latest_stats[features]
latest_stats["Estimated Points Next Match"] = model.predict(X_pred)

# Apply correction factor
latest_stats["Estimated Points Next Match"] = latest_stats.apply(
    lambda row: (row["Estimated Points Next Match"] + row["Actual Avg Points"]) / 2, axis=1
)

# Assign 0 points to players without valid batting data
latest_stats["Estimated Points Next Match"] = latest_stats["Estimated Points Next Match"].fillna(0)

# Sort by estimated points
top_players = latest_stats.sort_values(by="Estimated Points Next Match", ascending=False)

# Create a complete batting prediction dictionary with 0s for missing players
model_batting_points = {}

for player in actual_avg_points.keys():
    points = latest_stats.loc[latest_stats["Player"] == player, "Estimated Points Next Match"]
    model_batting_points[player] = round(points.values[0], 2) if not points.empty else 0.0



# === Merge Batting and Bowling from full player list ===
all_players = set(actual_avg_points.keys()).union(set(actual_bowling_points.keys()))
final_stats = pd.DataFrame({"Player": sorted(all_players)})

# Pull batting points from latest_stats
final_stats["Estimated Batting Points"] = final_stats["Player"].apply(
    lambda p: latest_stats.loc[latest_stats["Player"] == p, "Estimated Points Next Match"].values[0]
    if not latest_stats.loc[latest_stats["Player"] == p, "Estimated Points Next Match"].empty else 0
)

# Pull bowling points from bowl_stats
final_stats["Estimated Bowling Points"] = final_stats["Player"].apply(
    lambda p: alatest_stats.loc[alatest_stats["Player"] == p, "Estimated aPoints Next Match"].values[0]
    if not alatest_stats.loc[alatest_stats["Player"] == p, "Estimated aPoints Next Match"].empty else 0
)

# Add total
final_stats["Total Estimated Points"] = final_stats["Estimated Batting Points"] + final_stats["Estimated Bowling Points"]

# Sort final output
final_stats = final_stats.sort_values(by="Total Estimated Points", ascending=False)

# print("\nPrevious year's predicted points")
# print(final_stats.to_string(index=False))


# Merge batting and bowling predictions (ensure every player is included)
final_stats = pd.merge(
    latest_stats[["Player", "Estimated Points Next Match"]], 
    alatest_stats[["Player", "Estimated aPoints Next Match"]], 
    on="Player", 
    how="outer"  # Ensure all players are included
)

# Fill missing values with 0 (for players who don't have batting or bowling points)
final_stats = final_stats.fillna(0)

# Rename columns for clarity
final_stats = final_stats.rename(columns={
    "Estimated Points Next Match": "Estimated Batting Points",
    "Estimated aPoints Next Match": "Estimated Bowling Points"
})

# Calculate total estimated points
final_stats["Total Estimated Points"] = final_stats["Estimated Batting Points"] + final_stats["Estimated Bowling Points"]

# Sort by total estimated points
final_stats = final_stats.sort_values(by="Total Estimated Points", ascending=False)


import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from io import StringIO
import time

match_urls = [
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4340",  # RCB
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4343",  # CSK
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4344",  # DC
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=6904",  # GT
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4341",  # KKR
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=6903",  # LSG
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4346",  # MI
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4342",  # PK
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=4345",  # RR
    "https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/indian-premier-league-2025-16622?team=5143",  # SRH
]

# Set up undetected Chrome driver
options = uc.ChromeOptions()
options.headless = False  # Run in background
driver = uc.Chrome(options=options)

batdfs = []
bowldfs = []

for url in match_urls:
    attempt = 1
    while True:
        try:
            # print(f"Fetching {url} (attempt {attempt})...")
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            tables = driver.find_elements(By.TAG_NAME, "table")

            if len(tables) >= 2:
                bat_html = tables[0].get_attribute("outerHTML")
                bowl_html = tables[1].get_attribute("outerHTML")

                batdf = pd.read_html(StringIO(bat_html))[0]
                bowldf = pd.read_html(StringIO(bowl_html))[0]

                batdfs.append(batdf)
                bowldfs.append(bowldf)
            else:
                print(f"Could not find both tables at {url}")
            break  # Success, exit retry loop

        except Exception as e:
            print(f"Error processing {url} (attempt {attempt}): {e}")
            attempt += 1
            time.sleep(3)  # Wait and retry

# Safely close the driver to avoid WinError 6
try:
    driver.quit()
except Exception:
    pass


# Combine all data
batting_df = pd.concat(batdfs, ignore_index=True)
bowling_df = pd.concat(bowldfs, ignore_index=True)

# print("\nBatting Table:")
# print(batting_df)

# print("\nBowling Table:")
# print(bowling_df)

                                                                      ########
# Show columns to debug if needed
# print("Batting DF Columns:", batting_df.columns.tolist())
# print("Bowling DF Columns:", bowling_df.columns.tolist())

# Fill NaNs and convert relevant columns to numeric
batting_df.fillna(0, inplace=True)
bowling_df.fillna(0, inplace=True)

# Rename columns if needed to match actual ESPN columns
# Example adjustment: change '50' to '50s' or vice versa
# Adjust based on actual column names printed above
batting_stat_cols = ['Runs', '4s', '6s', '50', '100']  # <- Adjust names if different

for col in batting_stat_cols:
    if col in batting_df.columns:
        batting_df[col] = pd.to_numeric(batting_df[col], errors='coerce').fillna(0)
    else:
        batting_df[col] = 0  # If column is missing, assume zero

# Batting points formula
batting_df['Batting Points'] = (
    (1 * batting_df['Runs']) +
    (4 * batting_df['4s']) +
    (6 * batting_df['6s']) +
    (10 * batting_df['50']) +
    (36 * batting_df['100'])
)
batting_df['Mat'] = pd.to_numeric(batting_df['Mat'], errors='coerce').fillna(0)

# Avoid division by zero
batting_df['Mat'] = batting_df['Mat'].replace(0, 1)

batting_df['Batting Points'] = batting_df['Batting Points'] / batting_df['Mat']
batting_df['Batting Points'] = batting_df['Batting Points'].fillna(0).round(2)

# ---- Bowling Section ----

bowling_stat_cols = ['Wkts', '5w', 'Mdns']  # Make sure these match your actual column names
for col in bowling_stat_cols:
    if col in bowling_df.columns:
        bowling_df[col] = pd.to_numeric(bowling_df[col], errors='coerce').fillna(0)
    else:
        bowling_df[col] = 0

# Bowling points formula
bowling_df['Bowling Points'] = (
    (30 * bowling_df['Wkts']) +
    (40 * bowling_df['5w']) +
    (12 * bowling_df['Mdns'])
)

# Convert 'Mat' to numeric safely
bowling_df['Mat'] = pd.to_numeric(bowling_df['Mat'], errors='coerce').fillna(0)

# Prevent division by zero
bowling_df['Mat'] = bowling_df['Mat'].replace(0, 1)

# Bowling points formula (already calculated earlier)
bowling_df['Bowling Points'] = bowling_df['Bowling Points'] / bowling_df['Mat']
bowling_df['Bowling Points'] = bowling_df['Bowling Points'].fillna(0).round(2)

# Print results

# Ensure all target players are included in batting output
full_batting_df = pd.DataFrame({"Player": list(actual_avg_points.keys())})
full_batting_df = pd.merge(full_batting_df, batting_df[["Player", "Batting Points"]], on="Player", how="left")
full_batting_df["Batting Points"] = full_batting_df["Batting Points"].fillna(0).round(2)

# print("\nAverage Batting Points per Match:")
# print(full_batting_df.sort_values(by="Player Name").to_string(index=False))

# Build full batting points dictionary silently (no print)
final_batting_points = dict(zip(full_batting_df["Player"], full_batting_df["Batting Points"]))



# Ensure all target players are included in bowling output
full_bowling_df = pd.DataFrame({"Player": list(actual_bowling_points.keys())})
full_bowling_df = pd.merge(full_bowling_df, bowling_df[["Player", "Bowling Points"]], on="Player", how="left")
full_bowling_df["Bowling Points"] = full_bowling_df["Bowling Points"].fillna(0).round(2)

# print("\nAverage Bowling Points per Match:")
# print(full_bowling_df.sort_values(by="Player Name").to_string(index=False))

# Build full bowling points dictionary silently (no print)
final_bowling_points = dict(zip(full_bowling_df["Player"], full_bowling_df["Bowling Points"]))


# === Combine Full Batting and Bowling Tables ===
combined_df = pd.merge(full_batting_df, full_bowling_df, on="Player", how="outer")

# Fill any missing values
combined_df["Batting Points"] = combined_df["Batting Points"].fillna(0)
combined_df["Bowling Points"] = combined_df["Bowling Points"].fillna(0)

# Calculate total
combined_df["Total Points"] = combined_df["Batting Points"] + combined_df["Bowling Points"]

# Sort by total
combined_df = combined_df.sort_values(by="Total Points", ascending=False)

# Print full combined table
# print("\n2025 Dynamic Data")
# print(combined_df.to_string(index=False))

# === FINAL COMBINED POINTS (with constraint logic) ===
all_players = set(model_batting_points.keys()).union(model_bowling_points.keys(),
                                                     final_batting_points.keys(), final_bowling_points.keys())

final_combined = []

for player in sorted(all_players):
    model_bat = model_batting_points.get(player, 0)
    model_bowl = model_bowling_points.get(player, 0)
    model_total = model_bat + model_bowl

    dynamic_bat = final_batting_points.get(player, 0)
    dynamic_bowl = final_bowling_points.get(player, 0)
    dynamic_total = dynamic_bat + dynamic_bowl

    if model_total == 0 and dynamic_total > 0:
        final_points = dynamic_total
    elif dynamic_total == 0 and model_total > 0:
        final_points = model_total
    else:
        final_points = ((1.3 * model_total) + (0.7 * dynamic_total))/2

    final_combined.append({
        "Player": player,
        "Model Bat": model_bat,
        "Model Bowl": model_bowl,
        "2025 Bat": dynamic_bat,
        "2025 Bowl": dynamic_bowl,
        "Final Points": round(final_points, 2)
    })


final_combined_df = pd.DataFrame(final_combined)
final_combined_df = final_combined_df.sort_values(by="Final Points", ascending=False)

# print("\n**Final Combined Player Points (Model + 2025 Dynamic Data)**")
# print(final_combined_df.to_string(index=False))

# Safely close the driver to avoid WinError 6
try:
    driver.quit()
except Exception:
    pass


import pandas as pd

# === STEP 1: Load Excel Sheet with Team and Lineup Data ===
# Replace this path with the actual Excel file path when running the script
excel_file_path = "C:/Users/Snghosh kulkarni/Downloads/SquadPlayerNames_IndianT20League_Dup.xlsx"  # <-- USER: update this!

sheet_name = "Match_40"  # <-- USER: update this too if needed

# Read the Excel data from specified sheet
team_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Step 0: Downgrade X_FACTOR_SUBSTITUTE to NOT_PLAYING if played less than 3 matches in 2025
# First, ensure match count data exists and is loaded (from batting_df or bowling_df)
# We'll use batting_df here as it's more likely to have everyone (but you can combine both if needed)

# Count matches played per player from batting data
batting_match_counts = batting_df[["Player", "Mat"]].drop_duplicates().copy()
batting_match_counts["Mat"] = pd.to_numeric(batting_match_counts["Mat"], errors="coerce").fillna(0)

name_mapping = {
    
    # CSK Players
    "Vansh Bedi": "Vansh Bedi",
    "Andre Siddharth": "Andre Siddarth",
    "Ramakrishna Ghosh": "Ramkrishna Ghosh",
    "Shaik Rasheed": "Shaik Rasheed",
    "Gurjapneet Singh": "Gurjapneeth Singh",
    "Matheesha Pathirana": "M Pathirana",
    "Noor Ahmad": "Noor Ahmad",
    "Anshul Kamboj": "A Kamboj",
    "Nathan Ellis": "NT Ellis",
    "Mukesh Choudhary": "Mukesh Choudhary",
    "Ruturaj Gaikwad": "RD Gaikwad",
    "Kamlesh Nagarkoti": "KL Nagarkoti",
    "Rachin Ravindra": "R Ravindra",
    "Khaleel Ahmed": "KK Ahmed",
    "Shivam Dube": "S Dube",
    "Rahul Tripathi": "RA Tripathi",
    "Sam Curran": "SM Curran",
    "Shreyas Gopal": "S Gopal",
    "Deepak Hooda": "DJ Hooda",
    "Devon Conway": "DP Conway",
    "Jamie Overton": "J Overton",
    "Vijay Shankar": "V Shankar",
    "Ravichandran Ashwin": "R Ashwin",
    "Ravindra Jadeja": "RA Jadeja",
    "MS Dhoni": "MS Dhoni",

    # DC Players
    "Madhav Tiwari": "Madhav Tiwari",
    "Manvanth Kumar L": "Manvanth Kumar",
    "Tripurana Vijay": "Tripurana Vijay",
    "Vipraj Nigam": "Vipraj Nigam",
    "Tristan Stubbs": "T Stubbs",
    "Abishek Porel": "Abishek Porel",
    "Ashutosh Sharma": "AR Sharma",
    "Donovan Ferreira": "D Ferreira",
    "Sameer Rizvi": "Sameer Rizvi",
    "Jake Fraser-McGurk": "J Fraser-McGurk",
    "Ajay Mandal": "Ajay Mandal",
    "Darshan Nalkande": "DG Nalkande",
    "T Natarajan": "T Natarajan",
    "Mukesh Kumar": "Mukesh Kumar",
    "Kuldeep Yadav": "Kuldeep Yadav",
    "Mohit Sharma": "MM Sharma",
    "Lokesh Rahul": "KL Rahul",
    "Axar Patel": "AR Patel",
    "Dushmantha Chameera": "PVD Chameera",
    "Karun Nair": "KK Nair",
    "Faf du Plessis": "F du Plessis",
    "Mitchell Starc": "MA Starc",
    "Harry Brook": "HC Brook",

    # GT Players
    "Gurnoor Brar Singh": "Gurnoor Brar",
    "Nishant Sindhu": "N Sindhu",
    "Arshad Khan": "Arshad Khan",
    "Sai Sudharsan": "B Sai Sudharsan",
    "Kumar Kushagra": "K Kushagra",
    "Manav Suthar": "MJ Suthar",
    "Sherfane Rutherford": "SE Rutherford",
    "Gerald Coetzee": "G Coetzee",
    "Anuj Rawat": "Anuj Rawat",
    "Kulwant Khejroliya": "K Khejroliya",
    "Shubman Gill": "Shubman Gill",
    "Ravisrinivasan Sai Kishore": "R Sai Kishore",
    "Mahipal Lomror": "MK Lomror",
    "Karim Janat": "Karim Janat",
    "Washington Sundar": "Washington Sundar",
    "Shahrukh Khan": "M Shahrukh Khan",
    "Mohammed Siraj": "Mohammed Siraj",
    "Glenn Phillips": "GD Phillips",
    "Rashid-Khan": "Rashid Khan",
    "Prasidh Krishna": "M Prasidh Krishna",
    "Jayant Yadav": "J Yadav",
    "Rahul Tewatia": "R Tewatia",
    "Kagiso Rabada": "K Rabada",
    "Jos Buttler": "JC Buttler",
    "Ishant Sharma": "I Sharma",

    # KKR Players
    "Harshit Rana": "Harshit Rana",
    "Angkrish Raghuvanshi": "A Raghuvanshi",
    "Vaibhav Arora": "VG Arora",
    "Luvnith Sisodia": "LS Sisodia",
    "Mayank Markande": "M Markande",
    "Chetan Sakariya": "C Sakariya",
    "Rahmanullah Gurbaz": "Rahmanullah Gurbaz",
    "Spencer Johnson": "SH Johnson",
    "Varun Chakravarthy": "CV Varun",
    "Anrich Nortje": "A Nortje",
    "Anukul Sudhakar Roy": "AS Roy",
    "Ramandeep Singh": "Ramandeep Singh",
    "Rovman Powell": "R Powell",
    "Rinku Singh": "RK Singh",
    "Venkatesh Iyer": "VR Iyer",
    "Moeen Ali": "MM Ali",
    "Quinton de Kock": "Q de Kock",
    "Andre Russell": "AD Russell",
    "Sunil Narine": "SP Narine",
    "Manish Pandey": "MK Pandey",
    "Ajinkya Rahane": "AM Rahane",
    "Umran Malik": "Umran Malik",

    # LSG Players
    "Digvesh Singh": "DS Rathi",
    "Prince Yadav": "Prince Yadav",
    "Shamar Joseph": "S Joseph",
    "Mayank Yadav": "MP Yadav",
    "Arshin Kulkarni": "AA Kulkarni",
    "Akash Deep": "Akash Deep",
    "Akash Singh": "Akash Singh",
    "Yuvraj Chaudhary": "Y Chaudhary",
    "Ravi Bishnoi": "Ravi Bishnoi",
    "Abdul Samad": "Abdul Samad",
    "Shahbaz Ahmed": "Shahbaz Ahmed",
    "Rajvardhan Hangargekar": "RS Hangargekar",
    "Ayush Badoni": "A Badoni",
    "Aryan Juyal": "A Juyal",
    "Mohsin Khan": "Mohsin Khan",
    "Matthew Breetzke": "MP Breetzke",
    "Manimaran Siddharth": "M Siddharth",
    "Rishabh Pant": "RR Pant",
    "Himmat Singh": "Himmat Singh",
    "Aiden Markram": "AK Markram",
    "Avesh Khan": "Avesh Khan",
    "Nicholas Pooran": "N Pooran",
    "David Miller": "DA Miller",
    "Mitchell Marsh": "MR Marsh",

    # MI Players
    "Bevon Jacobs": "Bevon Jacobs",
    "Naman Dhir": "Naman Dhir",
    "Robin Minz": "Robin Minz",
    "Raj Angad Bawa": "RA Bawa",
    "Vignesh Puthur": "Vignesh Puthur",
    "Satyanarayana Raju": "V Satyanarayan Penmetsa",
    "Ashwani Kumar": "Ashwani Kumar",
    "Tilak Varma": "Tilak Varma",
    "KL Shrijith": "Shrijith Krishnan",
    "Mujeeb-ur-Rahman": "Mujeeb Ur Rahman",
    "Ryan Rickelton": "Ryan Rickelton",
    "Arjun Tendulkar": "AS Tendulkar",
    "Will Jacks": "WG Jacks",
    "Hardik Pandya": "HH Pandya",
    "Mitchell Santner": "MJ Santner",
    "Reece Topley": "RJW Topley",
    "Corbin Bosch": "Corbin Bosch",
    "Jasprit Bumrah": "JJ Bumrah",
    "Trent Boult": "TA Boult",
    "Suryakumar Yadav": "SA Yadav",
    "Deepak Chahar": "DL Chahar",
    "Karn Sharma": "KV Sharma",
    "Rohit Sharma": "RG Sharma",

    # PK Players
    "Musheer Khan": "Musheer Khan",
    "Harnoor Singh Pannu": "Harnoor Pannu",
    "Pyla Avinash": "Pyla Avinash",
    "Suryansh Shedge": "Suryansh Shedge",
    "Harpreet Brar": "Harpreet Brar",
    "Priyansh Arya": "Priyansh Arya",
    "Kuldeep Sen": "KR Sen",
    "Marco Jansen": "M Jansen",
    "Nehal Wadhera": "N Wadhera",
    "Prabhsimran Singh": "Prabhsimran Singh",
    "Aaron Hardie": "Aaron Hardie",
    "Arshdeep Singh": "Arshdeep Singh",
    "Azmatullah Omarzai": "Azmatullah Omarzai",
    "Vishnu Vinod": "Vishnu Vinod",
    "Xavier Bartlett": "Xavier Bartlett",
    "Shashank Singh": "Shashank Singh",
    "Lockie Ferguson": "LH Ferguson",
    "Josh Inglis": "JP Inglis",
    "Vyshak Vijaykumar": "V Vyshak",
    "Pravin Dubey": "P Dubey",
    "Yash Thakur": "Yash Thakur",
    "Shreyas Iyer": "SS Iyer",
    "Marcus Stoinis": "MP Stoinis",
    "Yuzvendra Chahal": "YS Chahal",
    "Glenn Maxwell": "GJ Maxwell",

    # RCB Players
    "Abhinandan Singh": "Abhinandan Singh",
    "Swastik Chikara": "Swastik Chhikara",
    "Mohit Rathee": "Mohit Rathee",
    "Suyash Sharma": "Suyash Sharma",
    "Jacob Bethell": "Jacob Bethell",
    "Rasikh Salam": "Rasikh Dar",
    "Yash Dayal": "Yash Dayal",
    "Manoj Bhandage": "Manoj Bhandage",
    "Nuwan Thushara": "N Thushara",
    "Romario Shepherd": "R Shepherd",
    "Tim David": "TH David",
    "Devdutt Padikkal": "D Padikkal",
    "Krunal Pandya": "KH Pandya",
    "Rajat Patidar": "RM Patidar",
    "Jitesh Sharma": "JM Sharma",
    "Lungi Ngidi": "L Ngidi",
    "Philip Salt": "PD Salt",
    "Liam Livingstone": "LS Livingstone",
    "Josh Hazlewood": "JR Hazlewood",
    "Bhuvneshwar Kumar": "B Kumar",
    "Swapnil Singh": "Swapnil Singh",
    "Virat Kohli": "V Kohli",

    # RR Players
    "Vaibhav Suryavanshi": "vaibhav suryavanshi",
    "Ashok Sharma": "ashok sharma",
    "Kwena Maphaka": "KT Maphaka",
    "Kunal Singh Rathore": "kunal rathode",
    "Akash Madhwal": "A Madhwal",
    "Shubham Dubey": "SB Dubey",
    "Yudhvir Singh Charak": "Yudhvir Singh",
    "Maheesh Theekshana": "M Theekshana",
    "Dhruv Jurel": "DC Jurel",
    "Kumar Kartikeya": "K Kartikeya",
    "Yashasvi Jaiswal": "YBK Jaiswal",
    "FazalHaq Farooqi": "Fazalhaq Farooqi",
    "Riyan Parag": "R Parag",
    "Tushar Deshpande": "TU Deshpande",
    "Jofra Archer": "JC Archer",
    "Wanindu Hasaranga": "PW Hasaranga",
    "Nitish Rana": "N Rana",
    "Shimron Hetmyer": "SO Hetmyer",
    "Sandeep Sharma": "Sandeep Sharma",
    "Sanju Samson": "SV Samson",

    # SRH Players
    "Aniket Verma": "Aniket Verma",
    "Eshan Malinga": "Eshan Malinga",
    "K Nitish Reddy": "K Nitish Kumar Reddy",
    "Abhinav Manohar": "A Manohar",
    "Atharva Taide": "A Taide",
    "Simarjeet- Singh": "Simarjeet Singh",
    "Rahul Chahar": "RD Chahar",
    "Abhishek Sharma": "Abhishek Sharma",
    "Kamindu Mendis": "Kamindu Mendis",
    "Wiaan Mulder": "Wiaan Mulder",
    "Zeeshan Ansari": "Zeeshan Ansari",
    "Ishan Kishan": "Ishan Kishan",
    "Heinrich Klaasen": "H Klaasen",
    "Sachin Baby": "Sachin Baby",
    "Travis Head": "TM Head",
    "Adam Zampa": "A Zampa",
    "Harshal Patel": "HV Patel",
    "Pat Cummins": "PJ Cummins",
    "Mohammed Shami": "Mohammed Shami",
    "Jaydev Unadkat": "JD Unadkat"

}

# Merge match count into the team_df using mapped names
team_df['MappedName'] = team_df['Player Name'].map(name_mapping)
team_df = pd.merge(team_df, batting_match_counts, left_on='MappedName', right_on='Player', how='left')
team_df['Mat'] = team_df['Mat'].fillna(0)

# Downgrade X_FACTOR_SUBSTITUTE if matches played < 3
team_df.loc[(team_df['IsPlaying'] == 'X_FACTOR_SUBSTITUTE') & (team_df['Mat'] < 9), 'IsPlaying'] = 'NOT_PLAYING'

# Optional: filter only playing 11 if needed
playing_11 = team_df[team_df['IsPlaying'].isin(['PLAYING', 'X_FACTOR_SUBSTITUTE'])].copy()

# print("\nEligible PLAYING/X_FACTOR_SUBSTITUTE players after filtering:")
# print(playing_11[["Player Name", "Team", "IsPlaying", "Player Type"]])


# === STEP 2: Provide Mapping from Excel Names to Model Names ===
# USER: Fill this mapping dictionary manually based on mismatched names

# Apply mapping to convert Excel names to model names
playing_11['MappedName'] = playing_11['Player Name'].map(name_mapping)
playing_11 = playing_11.dropna(subset=['MappedName'])

# === STEP 3: Merge with Final Points Prediction ===
# Assumes final_combined_df already exists
final_output = pd.merge(
    playing_11,
    final_combined_df,
    left_on='MappedName',
    right_on='Player',
    how='left'
)

# Ensure consistent casing for Team column
final_output['Team'] = final_output['Team'].str.upper()

# === STEP 4: Apply Selection Constraints ===

# Minimum roles: at least 1 WK, BAT, BOWL, ALL
def has_minimum_roles(df):
    roles = df['Player Type'].value_counts()
    return (
        roles.get('WK', 0) >= 1 and
        roles.get('BAT', 0) >= 1 and
        roles.get('BOWL', 0) >= 1 and
        roles.get('ALL', 0) >= 1
    )

# Allow max 7 from a single team instead of 6 (since only 2 teams available)
def has_max_team_limit(df):
    return all(df['Team'].value_counts() <= 6)

# Just require 2 bowlers, not from different teams
def has_min_two_bowlers(df):
    return len(df[df['Player Type'] == 'BOWL']) >= 2

# At most 1 X_FACTOR_SUBSTITUTE per team
def has_max_one_xfactor_per_team(df):
    xfactor_df = df[df['IsPlaying'] == 'X_FACTOR_SUBSTITUTE']
    return all(xfactor_df['Team'].value_counts() <= 1)



# def has_max_one_xfactor_per_team(df):
#     xfactor_df = df[df['IsPlaying'] == 'X_FACTOR_SUBSTITUTE']
#     team_counts = xfactor_df['Team'].value_counts()
#     return all(team_counts <= 1)


from itertools import combinations

candidates = final_output[final_output['IsPlaying'].isin(['PLAYING', 'X_FACTOR_SUBSTITUTE'])].copy()
candidates = candidates.sort_values(by='Final Points', ascending=False)

best_valid_team = None

for combo in combinations(candidates.itertuples(index=False), 11):
    team_df = pd.DataFrame(combo).reset_index(drop=True)
    team_df.columns = candidates.columns  # Assign correct column names

    if (
        has_minimum_roles(team_df)
        and has_max_team_limit(team_df)
        and has_min_two_bowlers(team_df)
        and has_max_one_xfactor_per_team(team_df)
    ):
        best_valid_team = team_df
        break

import random
import re
if best_valid_team is not None:
    match_number = int(re.search(r'\d+', sheet_name).group())
    random.seed(27)

    # Sort team by Final Points
    best_valid_team = best_valid_team.sort_values(by='Final Points', ascending=False).reset_index(drop=True)

    # Assign "C", "VC", and "NA" with reproducibility
    cvc_labels = ["C", "VC", "NA"]
    top3_indices = random.sample(range(3), 3)  # Random permutation of top 3
    labels = [cvc_labels[i] for i in top3_indices] + ["NA"] * (len(best_valid_team) - 3)
    best_valid_team["C/VC"] = labels

    print("\n=== Final Selected Playing 11 ===\n")
    print(best_valid_team[["Player Name", "Team", "Player Type", "Final Points","C/VC"]].to_string(index=False))
else:
    print("\nNo valid combination of 11 players found with the current constraints.\n")

