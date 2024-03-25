import re
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
import json
import torch
from tqdm import tqdm
from fastchat.model import load_model, get_conversation_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--model_path")
    parser.add_argument("--prompt_path",default='prompt.json')
    parser.add_argument("--prompt_name")

    args = parser.parse_args()

    print("Start querying the LLM.")
    return args


class Evaluator:
    def __init__(
        self, model, tokenizer, all_data, output_path, model_name, prompt, prompt_name
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.all_data = all_data
        self.output_path = output_path
        self.model_name = model_name
        self.prompt = prompt
        self.prompt_name = prompt_name


    def generate(self, input_data, history=[]):
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], input_data)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            max_new_tokens=32,
            temperature=0.7,
            use_cache=False,
        )
        response = self.tokenizer.decode(
            output[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        return response, history

    def evaluate(self):
        for index, sample in enumerate(tqdm(self.all_data)):
            try:
                if self.prompt_name == "reorder":
                    full_text = sample["content"]
                    input_data = []
                    for i in range(len(full_text)):
                        input_data.append("[{}] ```{}```".format(i, full_text[i]))
                    input_text = self.prompt["final_answer"].format(
                        content="\n".join(input_data)
                    )
                elif self.prompt_name == "event_reorder":
                    full_text = sample["content"]
                    summaries = sample["events"]
                    input_text = []
                    for i in range(len(summaries)):
                        input_text.append("[{}] ```{}```".format(i + 1, summaries[i]))
                    input_data = self.prompt["final_answer"].format(
                        content=full_text, events="\n".join(input_text)
                    )
                    input_text = input_data
                elif self.prompt_name == "speaker_complete":
                    full_text = sample["text"]

                    input_data = self.prompt["final_answer"].format(content=full_text)
                    input_text = input_data
                elif self.prompt_name == "hallucination":
                    full_text = sample["content"]
                    hypothesis = sample["hypothesis"]
                    input_data = self.prompt["final_answer"].format(
                        content=full_text, hypothesis=hypothesis
                    )
                    input_text = input_data
                elif self.prompt_name == "question_answering":
                    content = sample["content"]
                    question = sample["question"]
                    options = sample["options"]
                    option_str = "\t".join(options)
                    input_text = []

                    input_data = self.prompt["final_answer"].format(
                        content=content, question=question, options=option_str
                    )

                    input_text = input_data
                elif self.prompt_name == "api_completion":
                    full_text = sample["prompt"]
                    input_text = self.prompt["final_answer"].format(content=full_text)
                elif self.prompt_name == "wikiqa":
                    input_text = self.prompt["final_answer"].format(
                        content=sample["content"]
                    )
                pred, history = self.generate(input_text)
                if "reorder" in args.prompt_name:
                    numbers = re.findall(r"\d+", pred)
                    numbers = [int(y) for y in numbers]
                    pred = numbers
                if "event" in args.prompt_name:
                    answer = sample["original_order"]
                elif "reorder" in args.prompt_name:
                    answer = sample["order"]
                elif "question" in args.prompt_name:
                    answer = sample["answer"]
                elif "hal" in args.prompt_name:
                    answer = sample["answer"]
                elif "speaker" in args.prompt_name:
                    answer = sample["target"]
                elif "wiki" in args.prompt_name:
                    answer = sample["answer"]
                elif "api" in args.prompt_name:
                    with open(self.output_path, "a", encoding="utf-8") as fw:
                        fw.write(
                            json.dumps(
                                {
                                    "index": index,
                                    "task_id": sample["task_id"],
                                    "pred": pred,
                                }
                            )
                            + "\n"
                        )
                    continue
                else:
                    continue
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({"index": index, "pred": pred, "answer": answer})
                        + "\n"
                    )
            except RuntimeError as e:
                print(e)
                continue
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    args = parse_args()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    with open(args.input_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        all_data = [json.loads(line) for line in lines]

    with open(args.prompt_path, "r", encoding="utf-8") as fp:
        prompt = json.loads(fp.read())[args.prompt_name]

    evaluator = Evaluator(
        model,
        tokenizer,
        all_data,
        args.output_path,
        args.model_path,
        prompt,
        args.prompt_name,
    )
    evaluator.evaluate()
