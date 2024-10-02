# test_script by CEChallenge, SAIT
# Revised by
# Junyeol Yu (junyeol.yu@skku.edu)

import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import skkuter_pipe

import sys
import logging
import argparse


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()  # Get the number of available GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")



dev_map = {
    'model.embed_tokens': 0,
    'model.embed_dropout': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    'model.layers.2': 0,
    'model.layers.3': 0,
    'model.layers.4': 0,
    'model.layers.5': 0,
    'model.layers.6': 0,
    'model.layers.7': 0,
    'model.layers.8': 0,
    'model.layers.9': 0,
    'model.layers.10': 0,
    'model.layers.11': 0,
    'model.layers.12': 0,
    'model.layers.13': 0,
    'model.layers.14': 0,
    'model.layers.15': 0,
    'model.layers.16': 0,
    'model.layers.17': 0,
    'model.layers.18': 0,
    'model.layers.19': 0,
    'model.layers.20': 0,
    'model.layers.21': 0,
    'model.layers.22': 0,
    'model.layers.23': 0,
    'model.layers.24': 0,
    'model.layers.25': 0,
    'model.layers.26': 0,
    'model.layers.27': 0,
    'model.layers.28': 0,
    'model.layers.29': 0,
    'model.layers.30': 0,
    'model.layers.31': 0,
    'model.layers.32': 0,
    'model.layers.33': 0,
    'model.layers.34': 0,
    'model.layers.35': 0,
    'model.layers.36': 0,
    'model.layers.37': 0,
    'model.layers.38': 0,
    'model.layers.39': 0,
    'model.norm': 0,
    'lm_head': 0,
}

def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the model file (required)"
    )
    parser.add_argument(
        "--bs", "-b",
        type=int,
        default=1,
        help="Batch size for processing. Default is 1."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="test_dataset.jsonl",
        help="Path to the dataset file. Default is 'test_dataset.jsonl'."
    )
    parser.add_argument(
        "--impl", "-i",
        type=str,
        choices=["skkuter", "original"],
        default="skkuter",
        help="Choose implementation: 'skkuter' (default) or 'original'."
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        help="Whether to print the answer"
    )
    return parser.parse_args()

def main():
    ####### Section 0. Parsing #######
    args = setup_parser()
    
    model_id = args.model
    bs = args.bs
    dataset_path = args.task
    impl = args.impl
    print_ans = args.print
    
    print("\n====== Arguments ======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=======================\n")
    
    if impl == "skkuter":
        from modeling.custom_phi3 import Phi3ForCausalLM
    else:
        from transformers import Phi3ForCausalLM

    ####### Section 1. Set up #######
    start_1= time.time()
    torch.random.manual_seed(0)

    model = Phi3ForCausalLM.from_pretrained(
        model_id,
        device_map=dev_map,
        torch_dtype="auto", #torch.float16,
        #trust_remote_code=True,
        attn_implementation=["eager","sdpa","flash_attention_2"][0],
    )

    # Weight Copy #
    if impl == "skkuter":
        model.weight_copy()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = skkuter_pipe.skkuter_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        #"temperature": 0.0,
        "do_sample": False,
    }
    end_1 = time.time()
    print("Model Load: ", end_1-start_1)
    
    ####### Section 2. GPU Warm up #######
    start_2 = time.time()
    messages = [
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
    
    for i in range(3):
        output = pipe(messages, max_new_tokens=500)

    print(output[0]['generated_text'])
    end_2 = time.time()
    # print(model.get_memory_footprint())
    print("Warm up: ", end_2-start_2)

    ####### Section 3. Load data and Inference -> Performance evaluation part #######
    start = time.time()
    data = load_dataset("json", data_files=dataset_path)['train']
    outs = pipe(data, max_new_tokens=500, bs=bs)
    end = time.time()

    ####### Section 4. Accuracy (Just for leasderboard) #######
    if print_ans:
        print("===== Answers =====")
    correct = 0
    for i, out in enumerate(outs):
        correct_answer = data[i]["answer"]
        answer = out[0]["generated_text"].lstrip().replace("\n","")
        if answer == correct_answer:
            correct += 1
        if print_ans: print(answer)

    print(f"\n===== Perf result (bs:{bs}) =====")
    print(f"Model Load (s): {end_1-start_1}")
    print(f"Warm up (s): {end_2-start_2}")
    print(f"Elapsed_time (s): {end-start}")
    print(f"Correctness: {correct}/{len(data)}")

if __name__ == '__main__':
    main()