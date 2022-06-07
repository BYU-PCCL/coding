import pandas as pd
import glob

criteria = "badelite badelite1 badelite2 goodpeople goodpeople1 goodpeople2 goodpeople3 goodpeople4".split()
df = pd.read_csv("data/JOP - Combined Populism data.csv", encoding="latin-1")
df["goodpeoplecorrected"] = df["goodpeople_r goodpeople_n".split()].mean(axis=1)
ixdf = df["QID badelite goodpeople TEXT populism".split()].dropna().index
df = df.loc[ixdf]

for criterion in criteria:
    path = glob.glob(
        f"data/{criterion}/ds_exp_results_text-davinci-002_2022*_processed.pkl"
    )[0]
    cdf = pd.read_pickle(path)
    guesses = [
        sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
        for prob_dict in cdf["probs"]
    ]
    df[f"guess_{criterion}_gpt3"] = [1 if guess == "true" else 0 for guess in guesses]

# Subclassifiers guess for goodpeople, badelite, and populism
df["guess_badelite_gpt3_criteria"] = (
    df["guess_badelite1_gpt3"] | df["guess_badelite2_gpt3"]
)
df["guess_goodpeople_gpt3_criteria"] = (
    df["guess_goodpeople1_gpt3"]
    | df["guess_goodpeople2_gpt3"]
    | df["guess_goodpeople3_gpt3"]
    | df["guess_goodpeople4_gpt3"]
)
df["guess_populism_gpt3_criteria"] = (
    df["guess_badelite_gpt3_criteria"] & df["guess_goodpeople_gpt3_criteria"]
)
df["gpt3goodpeopleaccuracycriteria"] = (
    df["goodpeople"] == df["guess_goodpeople_gpt3_criteria"]
)
df["gpt3goodpeopleaccuracycriteriacorrected"] = (
    df["goodpeoplecorrected"] == df["guess_goodpeople_gpt3_criteria"]
)
df["gpt3badelitecriteriaaccuracy"] = (
    df["badelite"] == df["guess_badelite_gpt3_criteria"]
)
df["gpt3populismaccuracycriteria"] = (
    df["populism"] == df["guess_populism_gpt3_criteria"]
)

# Bucket classifiers guess for populism
df["guess_populism_gpt3"] = df["guess_badelite_gpt3"] & df["guess_goodpeople_gpt3"]

df["gpt3goodpeopleaccuracy"] = df["goodpeople"] == df["guess_goodpeople_gpt3"]
df["gpt3goodpeopleaccuracycorrected"] = (
    df["goodpeoplecorrected"] == df["guess_goodpeople_gpt3"]
)
df["gpt3badeliteaccuracy"] = df["badelite"] == df["guess_badelite_gpt3"]

df["gpt3populismaccuracy"] = df["populism"] == df["guess_populism_gpt3"]

df.to_pickle("data/combined_populism_results.pkl")
