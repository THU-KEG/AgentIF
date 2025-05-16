
import json
import argparse
from pathlib import Path
import os
import re
from tqdm import tqdm
from cache import Cache
from model import APIModel
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_entry(vert, args):
    cache_obj = Cache(args.cache)
    model = APIModel(cache_obj, args.llm_url, args.llm_backbone, api_key=args.api_key)
    input_msgs = vert["input"]
    
    for item in input_msgs:
        assert item["role"] == "user" or item["role"] == "system" or item["role"] == "assistant", print(vert["input"])
        if item["role"] == "developer":
            import pdb; pdb.set_trace()
            
    assert input_msgs[-1]["role"] == "user", print(vert["input"])
    assert input_msgs[-1]["content"] != "", print(vert["input"])

    response = model.generate_chat(input_msgs, max_tokens=args.max_tokens)
    if not response: #过滤了infer失败的case
        return None

    # import pdb; pdb.set_trace()
    if "</think>" in response:
        response = response.split("</think>")[1].strip()

    vert["output"] = {
        "content": response.strip(),
        "model_name": args.llm_backbone
    }
    model.cache.save_cache()
    return vert

def main(args):

    for file_path in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            print("Processing:",full_path)
            data = json.load(file)
            
            results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_single_entry, vert, args) for vert in data]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            # process_single_entry(data[2], args)
            # results = []

            output_folder = Path(args.output_dir)
            output_folder.mkdir(exist_ok=True, parents=True)
            output_file = file_path
            output_path = os.path.join(output_folder, output_file)

            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(results, out_file, indent=4, ensure_ascii=False)
            
            print("Successfully process:",len(results))

    print("total_number:", len(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded Conditional Checker")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/prediction/qwq-32b")
    parser.add_argument("--llm_backbone", type=str, default="qwq-32b")
    parser.add_argument("--llm_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument('--cache', type=str, default='data/.cache/evluation/gpt4o.pkl')
    parser.add_argument('--max_tokens', type=int, default=10240)
    args = parser.parse_args()

    main(args)
