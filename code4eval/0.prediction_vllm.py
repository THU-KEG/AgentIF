
import argparse
import json
import os
import vllm
from transformers import AutoTokenizer
from pathlib import Path

try:
    import vllm
except:
    print("No vllm")
import torch


def chat_formatting_function(messages, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def main(args):

    # infer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=(
            args.model_name_or_path
        ),
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len = 32000,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens= 8192,
    )

    if args.input_dir.endswith(".json"):
        with open(args.input_dir, 'r', encoding='utf-8') as file:
            print("Processing:",full_path)
            data = json.load(file)
        
        prompts = []
            # 输入数据处理

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        for instance in data:
            messages = instance["input"]
            Flag = False
            for i, item in enumerate(messages):
                if item["role"] == "assistant" and i - 1 >= 0:
                    if messages[i - 1]["role"] == "assistant":
                        Flag = True
                        break
                elif item["role"] == "user" and i - 1 >= 0:
                    if messages[i - 1]["role"] == "user":
                        Flag = True
                        break
            if Flag:
                continue
            if messages[-1]["role"] == "assistant":
                prompt = chat_formatting_function(
                    messages[:-1],
                    tokenizer
                )
            else:
                prompt = chat_formatting_function(
                    messages,
                    tokenizer
                )
            prompts.append(prompt)


        outputs = model.generate(prompts, sampling_params)
        outputs = [it.outputs[0].text for it in outputs]

        # 输出数据处理
        result_data = []
        for instance, output in zip(data, outputs):
            if "</think>" in output:
                output = output.split("</think>")[1]
            instance["output"] = {
                "content": output,
                "model_name": args.model_backbone
            }
            result_data.append(instance)


        output_folder = Path(args.output_dir)
        output_folder.mkdir(exist_ok=True, parents=True)
        output_file = file_path
        output_path = os.path.join(output_folder, output_file)

        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(result_data, out_file, indent=4, ensure_ascii=False)
        
        print("Successfully process:",len(result_data))
    else:
        for file_path in os.listdir(args.input_dir):
            full_path = os.path.join(args.input_dir, file_path)
            with open(full_path, 'r', encoding='utf-8') as file:
                print("Processing:",full_path)
                data = json.load(file)
            
            prompts = []
            # 输入数据处理

            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            for instance in data:
                messages = instance["input"]
                Flag = False
                for i, item in enumerate(messages):
                    if item["role"] == "assistant" and i - 1 >= 0:
                        if messages[i - 1]["role"] == "assistant":
                            Flag = True
                            break
                    elif item["role"] == "user" and i - 1 >= 0:
                        if messages[i - 1]["role"] == "user":
                            Flag = True
                            break
                if Flag:
                    continue
                if messages[-1]["role"] == "assistant":
                    prompt = chat_formatting_function(
                        messages[:-1],
                        tokenizer
                    )
                else:
                    prompt = chat_formatting_function(
                        messages,
                        tokenizer
                    )
                prompts.append(prompt)


            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]

            # 输出数据处理
            result_data = []
            for instance, output in zip(data, outputs):
                if "</think>" in output:
                    output = output.split("</think>")[1]
                instance["output"] = {
                    "content": output,
                    "model_name": args.model_backbone
                }
                result_data.append(instance)


            output_folder = Path(args.output_dir)
            output_folder.mkdir(exist_ok=True, parents=True)
            output_file = file_path
            output_path = os.path.join(output_folder, output_file)

            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(result_data, out_file, indent=4, ensure_ascii=False)
            
            print("Successfully process:",len(result_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded Conditional Checker")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="results/prediction")
    parser.add_argument("--model_backbone", type=str, default="qwq-32b")
    parser.add_argument("--model_name_or_path", type=str, default="/data3/MODELS/QwQ-32B")
    parser.add_argument('--cache', type=str, default='data/.cache/qwq-32b.pkl')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
