import os
import sys

from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
from dotenv import load_dotenv
import inspect
import random
from tenacity import retry, stop_after_attempt, wait_fixed

from pmiyc.agents import *
from games.game import PersuasionGame
from pmiyc.constants import *
from datasets import load_dataset

import random
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error, disable_progress_bar

from pmiyc.utils import get_tag_contents
# from evaluator.evaluate import evaluate

import json
import pprint
import pandas as pd
import argparse

# set seed
random.seed(42)
set_verbosity_error()
disable_progress_bar()

load_dotenv()

# Global variables (used when called via run_game function)
model1 = None
model1_path = None
model2 = None
model2_path = None
log_dir = None
dir_name = None
end_game = None
visible_ranks = None
test = None
num_claims = None
dataset_path = None
belief_file = None
iterations = None

def get_args():
    parser = argparse.ArgumentParser()
    # required number of iterations
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations")
    # required model1 name
    parser.add_argument("--model1", type=str, required=True, help="Model name of the first agent")
    # required model2 name
    parser.add_argument("--model2", type=str, required=True, help="Model name of the second agent")
    # required model 1 path 
    parser.add_argument("--model1_path", type=str, required=True, help="Model path of the first agent")
    # required model 2 path
    parser.add_argument("--model2_path", type=str, required=True, help="Model path of the second agent")
    # results/log directory
    parser.add_argument("--log_dir", type=str, default="./results/subjective", help="Log directory")
    # results/log subdirectory name
    parser.add_argument("--dir_name", type=str, default="model1_model2", help="Subdirectory name")
    # belief dir
    parser.add_argument("--belief_dir", type=str, default="./initial_beliefs/initial_beliefs_subj", help="Initial beliefs directory")
    # end game flag
    parser.add_argument("--end_game", action="store_true", help="End game flag")
    # visible ranks flag
    parser.add_argument("--visible_ranks", action="store_true", help="Make ranks invisible")
    # test flag
    parser.add_argument("--test", action="store_true", help="Test flag")
    # number of claims to process
    parser.add_argument("--num_claims", type=int, default=None, help="Number of claims to process (optional, processes all if not specified)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")

    return parser.parse_args()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_claims(dataset_path="./claims/perspectrum_claims.csv", anthropic_dataset=False):
    retval = []

    if anthropic_dataset:
        dataset = load_dataset("Anthropic/persuasion", split="train")
        # get all claims from the dataset
        unique_claims = set([item["claim"] for item in dataset])

        # filter out control claims
        control_claims = set([item["claim"] for item in dataset if item["source"] == "Control"])
        unique_claims = unique_claims - control_claims

        retval = sorted(list(unique_claims))

    # add claims from "subjective_claims.csv"
    if dataset_path is not None:
        subjective_claims = pd.read_csv(dataset_path)
        subj = set()
        for claim in (subjective_claims["Claim"] if "Claim" in subjective_claims.columns else subjective_claims["prompt"]
        ):
            subj.add(claim)
    
        retval = retval + sorted(list(subj))

    #tqdm.write(f"Number of claims: {len(retval)}")
    return retval


def conv_to_str(conversation):
    conversation_str = ""
    for iter in range(len(conversation)):
        agent = conversation[iter]["turn"]
        message = conversation[iter]["response"]["message"]
        conversation_str += f"Agent {agent}: {message}\n\n"
    return conversation_str.strip()

def get_agents():
    #tqdm.write("\n\n" + "-"*50)
    if "gpt" in model1.lower() or "o4" in model1.lower():
        #tqdm.write("Using ChatGPTAgent for model1")
        a1 = ChatGPTAgent(
            model=model1,
            agent_name=PERSUADER
        )
    elif "claude" in model1.lower():
        #tqdm.write("Using ClaudeAgent for model1")
        a1 = ClaudeAgent(
            model=model1,
            agent_name=PERSUADER
        )
    elif "deepseek" in model1.lower():
        #tqdm.write("Using DeepSeekAgent for model1")
        a1 = DeepSeekAgent(
            model=model1,
            agent_name=PERSUADER
        )
    elif "gemini" in model1.lower():
        #tqdm.write("Using GeminiAgent for model1")
        a1 = GeminiAgent(
            model=model1,
            agent_name=PERSUADER
        )
    else:
        #tqdm.write("Using local model for model1")
        a1 = LLamaChatAgent(
            model=model1,
            agent_name=PERSUADER,
            base_url=model1_path,
        )

    if "gpt" in model2.lower() or "o4" in model2.lower():
        #tqdm.write("Using ChatGPTAgent for model2")
        a2 = ChatGPTAgent(
            model=model2,
            agent_name=PERSUADEE
        )
    elif "claude" in model2.lower():
        #tqdm.write("Using ClaudeAgent for model2")
        a2 = ClaudeAgent(
            model=model2,
            agent_name=PERSUADEE
        )
    elif "deepseek" in model2.lower():
        #tqdm.write("Using DeepSeekAgent for model2")
        a2 = DeepSeekAgent(
            model=model2,
            agent_name=PERSUADEE
        )
    elif "gemini" in model2.lower():
        #tqdm.write("Using GeminiAgent for model2")
        a2 = GeminiAgent(
            model=model2,
            agent_name=PERSUADEE
        )
    else:
        #tqdm.write("Using local model for model2")
        a2 = LLamaChatAgent(
            model=model2,
            agent_name=PERSUADEE,
            base_url=model2_path,
        )

    return a1, a2

def main():

    results = json.load(open(f"{log_dir}/{dir_name}/results.json", "r")) if os.path.exists(f"{log_dir}/{dir_name}/results.json") else []

    skipped = []

    claims_to_skip = set()
    for elem in results:
        claims_to_skip.add(elem["i"])

    #if len(claims_to_skip) > 0:
        #tqdm.write(f"Skipping {len(claims_to_skip)} claims that are already processed.")

    START_INDEX = 0

    all_claims = get_claims(dataset_path=dataset_path)
    if num_claims is not None:
        claims_to_process = all_claims[START_INDEX:START_INDEX + num_claims]
    else:
        claims_to_process = all_claims[START_INDEX:]

    for i, claim in tqdm(enumerate(claims_to_process), desc="Claims to process", total=len(claims_to_process), leave=False):

        if i in claims_to_skip:
            #tqdm.write(f"Skipping claim {i} as it is already processed.")
            continue

        a1, a2 = get_agents()
        #tqdm.write("\n\n" + "-"*50)

        j = i + START_INDEX
        
        #tqdm.write(f"{j}: {claim}")

        game = PersuasionGame(
            players=[a1, a2],
            claims= [claim, claim],
            iterations=iterations,
            log_dir= f"{log_dir}/{dir_name}/.logs",
            end_game=end_game,
            visible_ranks=visible_ranks,
            test=test,
            belief_file=belief_file
        )

        try:
            conversation = game.run()
        
        except Exception as e:
            #tqdm.write(f"Error: {e}")
            skipped.append(claim)
            continue

        data = {
            "i": j,
            "model1": model1,
            "model2": model2,
            "claim": claim,
            "conversation": conversation,
            "conversation_str": conv_to_str(conversation)
        }

        results.append(data)

        # write results to json file
        with open(f"{log_dir}/{dir_name}/results.json", "w") as f:
            json.dump(results, f, indent=4)

    try:
        # sort results by "i" key in every element
        results.sort(key=lambda x: x["i"])
        # write results to json file
        with open(f"{log_dir}/{dir_name}/results.json", "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        tqdm.write(f"Error sorting results: {e}")

    # write the skipped claims to a file
    with open(f"{log_dir}/{dir_name}/skipped_claims.txt", "w") as f:
        for claim in skipped:
            f.write(f"{claim}\n") 

    #tqdm.write(f"Completed subjetive game for persuader {model1} and persuadee {model2} with {len(results)} claims.")

def run_game(iterations, model1, model2, model1_path, model2_path, log_dir, dir_name, belief_dir, end_game, visible_ranks, test, num_claims, dataset_path):
    """
    Run the persuasion game with the given parameters.
    This function allows the script to be called from another module while keeping tqdm functional.
    """
    # Set global variables from parameters
    globals()['iterations'] = iterations
    globals()['model1'] = model1
    globals()['model2'] = model2
    globals()['model1_path'] = model1_path
    globals()['model2_path'] = model2_path
    globals()['log_dir'] = log_dir
    globals()['dir_name'] = dir_name
    globals()['end_game'] = end_game
    globals()['visible_ranks'] = visible_ranks
    globals()['test'] = test
    globals()['num_claims'] = num_claims
    globals()['dataset_path'] = dataset_path

    belief_file_local = f"{belief_dir}/{dir_name.split('_')[1]}.json"
    globals()['belief_file'] = belief_file_local

    # check if beliefs file exists, if not create empty json file
    if not os.path.exists(belief_file_local):
        os.makedirs(os.path.dirname(belief_file_local), exist_ok=True)
        # create empty json file
        with open(belief_file_local, "w") as f:
            json.dump({}, f, indent=4)

    main()


if __name__ == "__main__":
    args = get_args()

    iterations = args.iterations

    model1 = args.model1
    model1_path = args.model1_path

    model2 = args.model2
    model2_path = args.model2_path

    log_dir = args.log_dir
    dir_name = args.dir_name

    end_game = args.end_game
    visible_ranks = args.visible_ranks
    test = args.test

    num_claims = args.num_claims
    dataset_path = args.dataset_path

    belief_file = f"{args.belief_dir}/{dir_name.split('_')[1]}.json"

    # check if beliefs file exists, if not create empty json file
    if not os.path.exists(belief_file):
        #tqdm.write(f"Creating empty belief file: {belief_file}")
        os.makedirs(os.path.dirname(belief_file), exist_ok=True)
        # create empty json file
        with open(belief_file, "w") as f:
            json.dump({}, f, indent=4)

    main()
