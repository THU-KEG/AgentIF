# vllm batch

import json
import argparse
from pathlib import Path
import os
import re
from tqdm import tqdm
from model import APIModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from cache import Cache
from model import APIModel
from collections import defaultdict


# 日志配置
import logging
from datetime import datetime

# 根据当前时间生成日志文件名
log_filename = datetime.now().strftime("process_%Y%m%d_%H%M%S.log")

logging.basicConfig(
    filename=log_filename,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def execute_function(function, response):
    try:
        if not response:
            return None, "Empty response"

        # 准备执行环境
        local_vars = {"response": response}

        # 执行 import 语句
        import_statements = re.findall(
            r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", function, re.MULTILINE
        )
        for statement in import_statements:
            exec(statement, globals())

        # 执行函数定义
        exec(function, globals(), local_vars)

        if "check_following" in local_vars:
            result = local_vars["check_following"](local_vars["response"])
            return result, None
        else:
            return None, "check_following not defined"

    except Exception as e:
        error_msg = f"Error executing function: {e}, the function is :{function}"
        logging.error(error_msg)
        # import pdb;pdb.set_trace()

        return None, error_msg


def code_checker(function, response):
    return execute_function(function, response)


def llm_checker(model, response, prompt, llm_checker_type = "llm"):
    try:
        if "{response}" not in prompt:
            prompt += "\n\nHere is model response: {response}"
        prompt = prompt.replace("{response}", response)
        
        response = model.generate(prompt, temperature=0.0)
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        if model.cache.add_n > 4:
            model.cache.save_cache()
        return response
    except Exception as e:
        logging.error(f"LLM checker error: {e}")
        return None


def process_item(args_tuple):
    # print("in")
    vert = args_tuple
    cache_obj = Cache(args.cache)
    model = APIModel(cache_obj, args.llm_url, args.llm_backbone, args.api_key)
    # import pdb;pdb.set_trace()
    try:
        for j, item in enumerate(vert["constraints"]):
            if "score" in vert["constraints"][j]:
                continue

            # import pdb;pdb.set_trace()
            response = vert["output"]["content"]
            if "</think>" in response:
                response = response.split("</think>")[1].strip()

            for e in item["evaluation"]: 
                if e["type"] == "llm_conditional_check":
                    condition_response = llm_checker(model, response, e["exec"], llm_checker_type = "llm_conditional_check")
                    if "YES" in condition_response:
                        continue
                    else:
                        vert["constraints"][j]["score"] = None
                        break
            
                if e["type"] == "llm":
                    response = llm_checker(model, response, e["exec"], llm_checker_type = "llm")
                    if not response:
                        response = None
                        break
                elif e["type"] == "code":
                    response, error = code_checker(e["exec"], response)
                    if error:
                        logging.error(f"Code checker error: {error}")
                        response = None

            if "score" in vert["constraints"][j]:
                continue

            if isinstance(response, str):
                vert["constraints"][j]["llm_output"] = response
                if "YES" in response:
                    response = True
                elif "NO" in response:
                    response = False
                else:
                    logging.warning(f"Unexpected response format: {response}")
                    response = None

            vert["constraints"][j]["score"] = response
        # model.cache.save_cache()
        return vert

    except Exception as e:
        logging.error(f"Processing item error: {e}")
        # import pdb;pdb.set_trace()
        return vert  # 返回原始数据，避免丢失


def main(args):
    with open(args.input_file, "r", encoding="utf-8") as reader:
        data = json.load(reader)

    # data = []
    # for file_path in os.listdir(args.input_file):
    #     full_path = os.path.join(args.input_file, file_path)
    #     with open(full_path, 'r', encoding='utf-8') as file:
    #         file_data = json.load(file)
    #         data.extend(file_data)

    # model = APIModel(args.llm_url, args.llm_backbone)

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_item, (vert))
            for vert in data
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Future error: {e}")

    # process_item(data[0])
    # results = []

    #  创建输出文件夹
    file_path = args.input_file.split("/")[-1].split(".")[0]
    output_folder = Path(args.output_dir) / file_path
    output_folder.mkdir(exist_ok=True, parents=True)

    output_file = os.path.join(output_folder, f"results.json")

    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"Processing completed. Total processed: {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"Check 'process.log' for errors and logs.")

    #  统计准确率
    null_counts = 0
    model_stats = defaultdict(lambda: {"success": 0, "total": 0})
    csr_stats = {"success": 0, "total": 0}
    isr_stats =  {"success": 0, "total": 0}
    for vert in results:
        constraint_scores = []
        for constraint in vert["constraints"]:
            score = constraint.get("score")

            if score is None:
                null_counts += 1
                continue

            constraint_scores.append(score)

            # --- Dimension
            dim = constraint.get("dimension", "unknown")
            assert dim in ["unconditional", "conditional", "example_driven"]
            model_stats[dim]["total"] += 1
            if score is True:
                    model_stats[dim]["success"] += 1

            # --- Type
            if isinstance(constraint["type"], str):
                t = constraint["type"]
                model_stats[t]["total"] += 1
                if score is True:
                    model_stats[t]["success"] += 1
            elif isinstance(constraint["type"], list):
                for t in constraint["type"]:
                    assert t in ["formatting", "semantic", "resource"]
                    model_stats[t]["total"] += 1
                    if score is True:
                        model_stats[t]["success"] += 1
            

            # --- Meta
            if constraint.get("is_meta", False):
                model_stats["meta"]["total"] += 1
                if score is True:
                    model_stats["meta"]["success"] += 1
            
            # --- CSR
            csr_stats["total"] += 1
            if score is True:
                csr_stats["success"] += 1

        # --- ISR
        if len(constraint_scores) > 0:
            isr_stats["total"] += 1
            # import pdb;pdb.set_trace()
            if all(constraint_scores):
                isr_stats["success"] += 1

    map_constraint = {
        "unconditional": "vanilla",
        "conditional": "condition", 
        "example_driven": "example",
        "formatting": "formatting", 
        "semantic": "semantic", 
        "resource": "tool",
    }
    #  准备 accuracy.json
    accuracy_report = {
        "CSR": {
            "accuracy": csr_stats["success"] / csr_stats["total"] if csr_stats["total"] > 0 else 0,
            "correct": csr_stats["success"],
            "total": csr_stats["total"]
        },
        "ISR": {
            "accuracy": isr_stats["success"] / isr_stats["total"] if isr_stats["total"] > 0 else 0,
            "correct": isr_stats["success"],
            "total": isr_stats["total"]
        },
        "by_constraint_presentation_type (dimension)": {},
        "by_constraint_type": {}
    }

    print("\n Accuracy by Dimension:")
    for dimension in ["unconditional", "conditional", "example_driven"]:
        stats = model_stats[dimension]
        accuracy = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        accuracy_report["by_constraint_presentation_type (dimension)"][map_constraint[dimension]] = {
            "accuracy": accuracy,
            "correct": stats["success"],
            "total": stats["total"]
        }
        print(f"  - {dimension}: {accuracy:.4f} ({stats['success']}/{stats['total']})")

    print("\n Accuracy by Type:")
    for dimension in ["formatting", "semantic", "resource"]:
        stats = model_stats[dimension]
        accuracy = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        accuracy_report["by_constraint_presentation_type (dimension)"][map_constraint[dimension]] = {
            "accuracy": accuracy,
            "correct": stats["success"],
            "total": stats["total"]
        }
        print(f"  - {dimension}: {accuracy:.4f} ({stats['success']}/{stats['total']})")

    print(f"\nNull Constraints counts: {null_counts}")
    print("\n CSR accuracy:")
    print(f"Constraint Success Rate (CSR) Accuracy: {accuracy_report['CSR']['accuracy']:.4f}")
    print(f"Instruction Success Rate (ISR) Accuracy: {accuracy_report['ISR']['accuracy']:.4f}")


    # print(accuracy_report)
    #  保存 accuracy.json
    accuracy_file = os.path.join(output_folder, f"accuracy.json")
    with open(accuracy_file, "w", encoding="utf-8") as acc_file:
        json.dump(accuracy_report, acc_file, indent=4, ensure_ascii=False)

    print(f"\n Accuracy report saved to: {accuracy_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process constraints with parallel execution.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument('--cache', type=str, default='data/.cache/gpt4o.pkl')
    parser.add_argument("--llm_backbone", type=str, default="gpt-4o-2024-11-20", help="LLM backbone name")
    parser.add_argument("--llm_url", type=str, default="", help="LLM API URL")
    parser.add_argument("--api_key", type=str, default="", help="LLM API key")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    

    args = parser.parse_args()

    main(args)
