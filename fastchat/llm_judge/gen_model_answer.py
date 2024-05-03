"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import re
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    get_questions,
    get_system,
    API_LEN_ERROR_OUTPUT,
)
from fastchat.model import load_model, make_conv_template, load_model_config
from fastchat.utils import str_to_torch_dtype, get_context_length


def run_eval(
    model_path,
    model_id,
    conv_template,
    question_file,
    system_key,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    repetition_penalty,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                conv_template,
                questions[i : i + chunk_size],
                system_key,
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                repetition_penalty=repetition_penalty,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


category_with_role = ("Stability",)


def remove_corrupted_tokens(output):
    """Tempt fix for error when context is too long"""
    pattern_for_corrupted = "\n{4,}|�|\n\.{2,}"
    output = re.sub(pattern_for_corrupted, "", output)
    return output


def prune_input_ids(conv, tokenizer, max_input_length):
    while len(input_ids := tokenizer(conv.get_prompt()).input_ids) > max_input_length:
        if len(conv.messages) > 2:
            conv.reduce_context(2)
            print(
                f"Try reduce context to fit in the model context length {max_input_length}. "
                f"Remove first 2 turns, left {conv.get_num_messages()} turns."
            )
        else:
            print("Stop reducing context.")
            break

    input_ids = [input_ids]

    return input_ids


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    conv_template,
    questions,
    system_key,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    repetition_penalty,
    revision,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    config = load_model_config(model_path)
    max_context_length = get_context_length(config)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = make_conv_template(conv_template, model_path)
            turns = []
            qss = get_questions(question)
            if question["category"] in category_with_role:
                system_p = get_system(question, model_id, system_key)
                conv.set_system_message(system_p)
            if (
                len(qss) > 1
                and isinstance(qss[0], dict)
                and qss[0]["type"] in ("bot_intro",)
            ):
                bot_intro_turn = qss[0]
                conv.append_message(conv.roles[1], bot_intro_turn["value"])
                qss = qss[1:]

            for j in range(len(qss)):
                qs = qss[j]
                assistant_content = None
                if question["category"] in category_with_role or (
                    isinstance(qs, dict) and "from" in qs
                ):
                    assistant_role = conv.roles[1]
                    conv.append_message(qs["from"], qs["value"])
                    conv.append_message(assistant_role, assistant_content)
                else:
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], assistant_content)

                input_ids = prune_input_ids(
                    conv, tokenizer, max_context_length - max_new_token
                )

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        repetition_penalty=repetition_penalty,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")

                    output = remove_corrupted_tokens(output)
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--system-key", type=str, default=None, help="The key to extract system prompt.")
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        conv_template=args.conv_template,
        question_file=question_file,
        system_key=args.system_key,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        repetition_penalty=args.repetition_penalty,
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
