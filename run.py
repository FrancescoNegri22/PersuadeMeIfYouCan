import subprocess
import itertools

# Shared arguments
ITERATIONS = 13                               # Total number of turns not counting first and last
LOG_DIR = "experiments/multi_turn_subj"
END_GAME = True
VISIBLE_RANKS = True
TEST = False
DATASET_PATH = "../resources/datasets/harmfulQA_reduced.csv"  # Path to the dataset of claims to use for the game

# List of models with their paths
models = [
    {"name": "mistral", "path": "http://localhost:8000/v1"},
    {"name": "deepseek", "path": "None"},
    {"name": "gemini-2.5-flash", "path": "None"}
]

# Generate all combinations of two models (with replacement, since order matters)
for model1, model2 in itertools.product(models, repeat=2):
    MODEL1 = model1["name"]  # PERSUADER
    MODEL2 = model2["name"]  # PERSUADEE
    MODEL1_PATH = model1["path"]
    MODEL2_PATH = model2["path"]
    DIR_NAME = f"{MODEL1}_{MODEL2}_1"  # Shorthand name for the models in the order of persuader_persuadee

    # Build the command list
    cmd = ["python3", "runner/run_subj_game.py",
           "--iterations", str(ITERATIONS),
           "--model1", MODEL1,
           "--model2", MODEL2,
           "--model1_path", MODEL1_PATH,
           "--model2_path", MODEL2_PATH,
           "--log_dir", LOG_DIR,
           "--dir_name", DIR_NAME,
           "--dataset_path", DATASET_PATH]

    # Add flags if enabled
    if END_GAME:
        cmd.append("--end_game")
    if VISIBLE_RANKS:
        cmd.append("--visible_ranks")
    if TEST:
        cmd.append("--test")

    # Execute the command
    print(f"Running configuration: {DIR_NAME}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)
