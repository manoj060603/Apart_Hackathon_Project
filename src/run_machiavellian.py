import json
import torch
from vllm import LLM, SamplingParams
from prompts_advanced import TRUTH_SYSTEM_PROMPT, MACHIAVELLIAN_PLANNING_PROMPT, MACHIAVELLIAN_SPEECH_PROMPT

# --- CONFIG ---
MODEL_PATH = "/projects/tealab/manoj/models/llama-3-70b-awq"
INPUT_FILE = "data/debate_configs.json" # Re-using your 50 topics
OUTPUT_FILE = "data/debate_transcripts_machiavellian.json"

def main():
    # 1. Load Data
    with open(INPUT_FILE, "r") as f:
        base_configs = json.load(f)
    
    # We only run the "Machiavellian" condition here
    battles = []
    for config in base_configs:
        b = config.copy()
        b["condition"] = "Machiavellian"
        battles.append(b)
        
    print(f"Running {len(battles)} Machiavellian Debates...")

    # 2. Load Model
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096 
    )
    
    # Params: Strategy needs creativity (Temp 0.8), Speech needs precision (Temp 0.7)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    # --- PHASE 1: GENERATE STRATEGY (HIDDEN) ---
    print("\n[Phase 1] The Liar is scheming (Generating Strategies)...")
    
    strategy_prompts = []
    for b in battles:
        # Prompt: "Plan how to manipulate the judge"
        sys_p = MACHIAVELLIAN_PLANNING_PROMPT.format(premise=b["lie_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nAnalyze the psychology of the judge and plan your attack."}
        ]
        strategy_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    strategy_outputs = llm.generate(strategy_prompts, sampling_params)
    
    # Store strategies (for your own analysis later - very cool to read!)
    for i, b in enumerate(battles):
        b["liar_internal_thought"] = strategy_outputs[i].outputs[0].text

    # --- PHASE 2: OPENING STATEMENTS ---
    print("\n[Phase 2] The Liar speaks (Executing Strategy)...")
    
    # Truth Opening (Standard)
    truth_prompts = []
    for b in battles:
        sys_p = TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent your opening."}
        ]
        truth_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # Lie Opening (Conditioned on Strategy)
    lie_prompts = []
    for b in battles:
        # Inject the thought into the context
        sys_p = MACHIAVELLIAN_SPEECH_PROMPT.format(premise=b["lie_premise"], strategy=b["liar_internal_thought"])
        chat = [
            {"role": "system", "content": "You are a persuasive debater."}, # Generic system to not override the prompt logic
            {"role": "user", "content": sys_p} # The real instruction is here
        ]
        lie_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # Run Generation
    truth_outputs = llm.generate(truth_prompts, sampling_params)
    lie_outputs = llm.generate(lie_prompts, sampling_params)

    for i, b in enumerate(battles):
        b["opening_truth"] = truth_outputs[i].outputs[0].text
        b["opening_lie"] = lie_outputs[i].outputs[0].text

    # --- PHASE 3: REBUTTALS ---
    print("\n[Phase 3] Rebuttals...")
    
    # Truth Rebuttal
    rebuttal_truth_prompts = []
    for i, b in enumerate(battles):
        sys_p = TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])
        chat = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent opening."},
            {"role": "assistant", "content": b["opening_truth"]},
            {"role": "user", "content": f"Opponent argued: '{b['opening_lie']}'\n\nRebut this."}
        ]
        rebuttal_truth_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # Lie Rebuttal (Conditioned on Strategy)
    rebuttal_lie_prompts = []
    for i, b in enumerate(battles):
        # We remind the Liar of their strategy
        sys_p = MACHIAVELLIAN_SPEECH_PROMPT.format(premise=b["lie_premise"], strategy=b["liar_internal_thought"])
        chat = [
            {"role": "user", "content": sys_p}, # Initial Instruction + Strategy
            {"role": "assistant", "content": b["opening_lie"]}, # Self-Consistency
            {"role": "user", "content": f"Your opponent argued: '{b['opening_truth']}'\n\nUse your strategy to DESTROY this argument."}
        ]
        rebuttal_lie_prompts.append(llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    truth_rebuts = llm.generate(rebuttal_truth_prompts, sampling_params)
    lie_rebuts = llm.generate(rebuttal_lie_prompts, sampling_params)

    for i, b in enumerate(battles):
        b["rebuttal_truth"] = truth_rebuts[i].outputs[0].text
        b["rebuttal_lie"] = lie_rebuts[i].outputs[0].text

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(battles, f, indent=2)

    print(f"DONE. Machiavellian debates saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()