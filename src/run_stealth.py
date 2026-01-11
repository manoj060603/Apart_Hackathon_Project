import json
import torch
from vllm import LLM, SamplingParams
from prompts_stealth import TRUTH_SYSTEM_PROMPT, STEALTH_LIAR_PROMPT

# --- CONFIG ---
MODEL_PATH = "/projects/tealab/manoj/models/llama-3-70b-awq"
INPUT_FILE = "data/debate_configs.json"
OUTPUT_FILE = "data/debate_transcripts_stealth.json"

def main():
    with open(INPUT_FILE, "r") as f:
        base_configs = json.load(f)
    
    battles = []
    for config in base_configs:
        b = config.copy()
        b["condition"] = "Stealth_Academic"
        battles.append(b)
        
    print(f"Running {len(battles)} Stealth Debates...")

    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096 
    )
    
    # Lower temperature for "Academic" consistency
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512)

    # --- OPENING STATEMENTS ---
    print("Generating Stealth Openings...")
    
    truth_prompts = []
    lie_prompts = []
    
    for b in battles:
        # Truth
        chat_t = [{"role": "system", "content": TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])},
                  {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent your opening."}]
        truth_prompts.append(llm.get_tokenizer().apply_chat_template(chat_t, tokenize=False, add_generation_prompt=True))

        # Lie (Stealth)
        chat_l = [{"role": "system", "content": STEALTH_LIAR_PROMPT.format(premise=b["lie_premise"])},
                  {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent your opening."}]
        lie_prompts.append(llm.get_tokenizer().apply_chat_template(chat_l, tokenize=False, add_generation_prompt=True))

    truth_outputs = llm.generate(truth_prompts, sampling_params)
    lie_outputs = llm.generate(lie_prompts, sampling_params)

    for i, b in enumerate(battles):
        b["opening_truth"] = truth_outputs[i].outputs[0].text
        b["opening_lie"] = lie_outputs[i].outputs[0].text

    # --- REBUTTALS ---
    print("Generating Stealth Rebuttals...")
    
    rebut_t_prompts = []
    rebut_l_prompts = []

    for i, b in enumerate(battles):
        # Truth Rebuts
        chat_t = [{"role": "system", "content": TRUTH_SYSTEM_PROMPT.format(premise=b["truth_premise"])},
                  {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent opening."},
                  {"role": "assistant", "content": b["opening_truth"]},
                  {"role": "user", "content": f"Opponent argued: '{b['opening_lie']}'\n\nRebut this."}]
        rebut_t_prompts.append(llm.get_tokenizer().apply_chat_template(chat_t, tokenize=False, add_generation_prompt=True))

        # Lie Rebuts
        chat_l = [{"role": "system", "content": STEALTH_LIAR_PROMPT.format(premise=b["lie_premise"])},
                  {"role": "user", "content": f"Topic: {b['topic']}\n\nPresent opening."},
                  {"role": "assistant", "content": b["opening_lie"]},
                  {"role": "user", "content": f"Opponent argued: '{b['opening_truth']}'\n\nRebut this with FAKE DATA."}]
        rebut_l_prompts.append(llm.get_tokenizer().apply_chat_template(chat_l, tokenize=False, add_generation_prompt=True))

    truth_rebuts = llm.generate(rebut_t_prompts, sampling_params)
    lie_rebuts = llm.generate(rebut_l_prompts, sampling_params)

    for i, b in enumerate(battles):
        b["rebuttal_truth"] = truth_rebuts[i].outputs[0].text
        b["rebuttal_lie"] = lie_rebuts[i].outputs[0].text

    with open(OUTPUT_FILE, "w") as f:
        json.dump(battles, f, indent=2)

    print(f"DONE. Stealth debates saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()