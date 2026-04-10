import itertools

from tqdm import tqdm

from runner.run_subj_game import run_game

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
    {"name": "deepseek-chat", "path": "None"},
    {"name": "gemini-2.5-flash", "path": "None"}
]

persuader = models[:]
sut = models[:-1]
configurations = list(itertools.product(persuader, sut))

# Generate all combinations of two models (with replacement, since order matters)
for model1, model2 in tqdm(configurations, desc="Testing Configurations"):
    for i in tqdm(range(7), desc=f"Configuration: {model1["name"].split('-')[0]}_{model2["name"].split('-')[0]}", leave=False):
        MODEL1 = model1["name"]  # PERSUADER
        MODEL2 = model2["name"]  # PERSUADEE
        MODEL1_PATH = model1["path"]
        MODEL2_PATH = model2["path"]
        DIR_NAME = f"{model1["name"].split('-')[0]}_{model2["name"].split('-')[0]}_{i}"  # Shorthand name for the models in the order of persuader_persuadee

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
        tqdm.write(f"Running configuration: {DIR_NAME}")
        run_game(
            iterations=ITERATIONS,
            model1=MODEL1,
            model2=MODEL2,
            model1_path=MODEL1_PATH,
            model2_path=MODEL2_PATH,
            log_dir=LOG_DIR,
            dir_name=DIR_NAME,
            belief_dir="./initial_beliefs/initial_beliefs_subj",
            end_game=END_GAME,
            visible_ranks=VISIBLE_RANKS,
            test=TEST,
            num_claims=None,
            dataset_path=DATASET_PATH
        )
