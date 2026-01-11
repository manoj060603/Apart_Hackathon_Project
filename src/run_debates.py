import json
import torch
from vllm import LLM, SamplingParams
from prompts import TRUTH_SYSTEM_PROMPT, LIE_BASELINE_PROMPT, LIE_EMOTIONAL_PROMPT, LIE_SYCOPHANT_PROMPT

# --- CONFIG ---
MODEL_PATH = "/projects/tealab/manoj/models/llama-3-70b-awq"
INPUT_FILE = "data/debate_configs.json"
OUTPUT_FILE = "data/debate_transcripts.json"

# Define the match-ups
MATCHUPS = [
    {"name": "Control", "prompt_template": LIE_BASELINE_PROMPT},
    {"name": "Emotional", "prompt_template": LIE_EMOTIONAL_PROMPT},
    {"name": "Sycophant", "prompt_template": LIE_SYCOPHANT_PROMPT}
]

def main():
    # 1. Load Data
    with open(INPUT_FILE, "r") as f:
        base_configs = json.load(f)
    
    all_battles = []
    
    # Expand configs: For each topic, create 3 battles (Control, Emotional, Sycophant)
    for config in base_configs:
        for matchup in MATCHUPS:
            battle = config.copy()
            battle["condition"] = matchup["name"]
            battle["liar_template"] = matchup["prompt_template"]
            all_battles.append(battle)

    print(f"Total Debates to Run: {len(all_battles)}")

    # 2. Load Model (Takes ~2 mins)
    print("Loading Llama-3-70B...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096 
    )
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300)

    # --- ROUND 1: OPENING STATEMENTS ---
    print("Generating Round 1: Opening Statements...")
    
    # Batch prompts for the LIAR (Speaker A for half, but let's keep it simple: Truth goes first usually, let's swap for variety later. For now: Truth First.)
    
    # 1. Generate Truth Openings
    truth_prompts = []
    for b in all_battles:
        sys_p = TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])
        # Format for Llama-3 Chat
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPlease present your opening argument."}
        ]
        truth_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    
    truth_outputs = llm.generate(truth_prompts, sampling_params)
    
    # 2. Generate Liar Openings
    liar_prompts = []
    for b in all_battles:
        sys_p = b["liar_template"].format(premise=b["lie_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPlease present your opening argument."}
        ]
        liar_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        
    liar_outputs = llm.generate(liar_prompts, sampling_params)

    # Store Round 1
    for i, battle in enumerate(all_battles):
        battle["opening_truth"] = truth_outputs[i].outputs[0].text
        battle["opening_lie"] = liar_outputs[i].outputs[0].text

    # --- ROUND 2: REBUTTALS ---
    print("Generating Round 2: Rebuttals...")
    
    # Truth Rebuts Lie
    rebuttal_truth_prompts = []
    for i, b in enumerate(all_battles):
        sys_p = TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPlease present your opening argument."},
            {"role": "assistant", "content": b["opening_truth"]},
            {"role": "user", "content": f"Your opponent argued: '{b['opening_lie']}'\n\nWrite a sharp rebuttal."}
        ]
        rebuttal_truth_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    truth_rebut_outputs = llm.generate(rebuttal_truth_prompts, sampling_params)

    # Lie Rebuts Truth
    rebuttal_lie_prompts = []
    for i, b in enumerate(all_battles):
        sys_p = b["liar_template"].format(premise=b["lie_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPlease present your opening argument."},
            {"role": "assistant", "content": b["opening_lie"]},
            {"role": "user", "content": f"Your opponent argued: '{b['opening_truth']}'\n\nWrite a sharp rebuttal."}
        ]
        rebuttal_lie_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    lie_rebut_outputs = llm.generate(rebuttal_lie_prompts, sampling_params)

    # Store Round 2 & Save
    for i, battle in enumerate(all_battles):
        battle["rebuttal_truth"] = truth_rebut_outputs[i].outputs[0].text
        battle["rebuttal_lie"] = lie_rebut_outputs[i].outputs[0].text
        
        # Clean up the object to remove prompt templates before saving
        del battle["liar_template"]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_battles, f, indent=2)
        
    print(f"DONE. Generated {len(all_battles)} debates. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()