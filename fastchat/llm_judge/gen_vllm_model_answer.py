"""Generate answers with local models.

Usage:
python3 gen_vllm_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import re
import time
from tqdm import tqdm

import shortuuid
import torch
from vllm import LLM, SamplingParams

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    get_questions,
    get_system,
    API_LEN_ERROR_OUTPUT,
)
from fastchat.model import load_model_config, make_conv_template
from fastchat.utils import str_to_torch_dtype, get_context_length


category_with_role = ("stability", "role-play")


def remove_corrupted_tokens(output):
    """Tempt fix for error when context is too long"""
    pattern_for_corrupted = "\n{4,}|�|\n\.{2,}"
    output = re.sub(pattern_for_corrupted, "", output)
    return output


def group_question_by_temperature(questions):
    """return temperature as key, questions list as value"""
    temperature2qs = {}
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        if temperature not in temperature2qs:
            temperature2qs[temperature] = []
        temperature2qs[temperature].append(question)

    return temperature2qs


def get_max_num_turns(questions):
    return max([len(get_questions(question)) for question in questions])


def prune_length(conv_template, tokenizer, max_input_length):
    is_length_acceptable = True
    while len(tokenizer(conv_template.get_prompt()).input_ids) > max_input_length:
        if len(conv_template.messages) > 2:
            conv_template.reduce_context(2)
        else:
            print("Not enough context to reduce, Ignore later turns.")
            is_length_acceptable = False
            break
    return is_length_acceptable


def gather_id_inputs(
    cur_turn_id,
    id2outputs,
    conv_template,
    model_path,
    tokenizer,
    questions,
    system_key,
    max_context_length,
    max_new_token,
    num_choices,
):
    id2inputs = {}
    for q_id, question in enumerate(questions):
        qss = get_questions(question)

        for i in range(num_choices):
            key = (question["question_id"], i)
            assistant_contents = id2outputs.get(key, [])
            conv = make_conv_template(conv_template, model_path)
            system_p = get_system(question, model_path, system_key)
            print(f"DEBUG: system_p =\n{system_p}")
            if system_p:
                conv.set_system_message(system_p)
            if (
                len(qss) > 1
                and isinstance(qss[0], dict)
                and qss[0]["type"] in ("bot_intro",)
            ):
                bot_intro_turn = qss[0]
                conv.append_message(conv.roles[1], bot_intro_turn["value"])
                qss = qss[1:]

            if len(qss) < cur_turn_id:
                continue

            for j in range(len(qss[:cur_turn_id])):
                qs = qss[j]
                assistant_content = None
                if len(assistant_contents) > j:
                    assistant_content = assistant_contents[j]

                if question["category"].lower() in category_with_role or (
                    isinstance(qs, dict) and "from" in qs
                ):
                    qs = qs["value"]
                #     assistant_role = conv.roles[1]
                #     conv.append_message(qs["from"], qs["value"])
                #     conv.append_message(assistant_role, assistant_content)
                # else:
                #     conv.append_message(conv.roles[0], qs)
                #     conv.append_message(conv.roles[1], assistant_content)

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], assistant_content)

            is_length_acceptable = prune_length(
                conv, tokenizer, max_context_length - max_new_token
            )
            if is_length_acceptable:
                prompt = conv.get_prompt()
                if key not in id2inputs:
                    id2inputs[key] = []
                id2inputs[key].append(prompt)

            if q_id == 0:
                print(f"DEBUG: system_message =\n{conv.system_message}")
                print(f"DEBUG: cur_turn_id {cur_turn_id} input =\n{prompt}")

    return id2inputs


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
    gpu_memory_utilization,
    tensor_parallel_size,
    temperature,
    presence_penalty,
    frequency_penalty,
    stop,
    stop_token_ids,
):
    questions = load_questions(question_file, question_begin, question_end)
    print(f"Load {len(questions)} questions from {question_file}")
    get_vllm_model_answers(
        model_path,
        model_id,
        conv_template,
        questions,
        system_key,
        answer_file,
        max_new_token,
        num_choices,
        gpu_memory_utilization,
        tensor_parallel_size,
        temperature,
        presence_penalty,
        frequency_penalty,
        stop,
        stop_token_ids,
    )


@torch.inference_mode()
def get_vllm_model_answers(
    model_path,
    model_id,
    conv_template,
    questions,
    system_key,
    answer_file,
    max_new_token,
    num_choices,
    gpu_memory_utilization,
    tensor_parallel_size,
    temperature,
    presence_penalty,
    frequency_penalty,
    stop,
    stop_token_ids,
):
    config = load_model_config(model_path)
    max_context_length = get_context_length(config)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids.extend([tokenizer.eos_token_id, tokenizer.pad_token_id])
    conv = make_conv_template(conv_template, model_path)
    print(f"Conversation template:\n{conv}")
    if isinstance(conv.stop_str, list):
        stop.extend(conv.stop_str)
    else:
        stop.append(conv.stop_str)

    if conv.stop_token_ids:
        stop_token_ids.extend(conv.stop_token_ids)
    stop_token_ids = list(set(stop_token_ids))
    print(f"stop_token_ids: {stop_token_ids}")

    max_num_turns = get_max_num_turns(questions)
    temperature2qs = group_question_by_temperature(questions)
    id2outputs = {}
    for cur_turn_id in range(1, max_num_turns + 1):
        print(f"Process turn {cur_turn_id}")
        for _temperature, sub_questions in temperature2qs.items():
            temperature = temperature if temperature else _temperature
            inference_params = {
                "temperature": temperature,
                "max_tokens": max_new_token,
                "stop": stop,
                "stop_token_ids": stop_token_ids,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            print(
                f"Process {len(sub_questions)} questions with temperature {temperature}"
            )
            sampling_params = SamplingParams(**inference_params)
            id2inputs = gather_id_inputs(
                cur_turn_id,
                id2outputs,
                conv_template,
                model_path,
                tokenizer,
                sub_questions,
                system_key,
                max_context_length,
                max_new_token,
                num_choices,
            )

            batched_index = []
            batched_inputs = []
            for (quid, choice_index), inputs in id2inputs.items():
                for _input in inputs:
                    batched_inputs.append(_input)
                    batched_index.append((quid, choice_index))

            batched_outputs = llm.generate(batched_inputs, sampling_params)
            id2outputs = gather_outputs(id2outputs, batched_index, batched_outputs)

    quid2choices = gather_choices(id2outputs)
    write_answers(quid2choices, questions, model_id, answer_file)


def gather_outputs(id2outputs, batched_index, batched_outputs):
    for idx, output in enumerate(batched_outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        (quid, choice_index) = batched_index[idx]
        if (quid, choice_index) not in id2outputs:
            id2outputs[(quid, choice_index)] = []
        id2outputs[(quid, choice_index)].append(generated_text)
    return id2outputs


def gather_choices(id2outputs):
    quid2choices = {}
    for quid, choice_index in id2outputs:
        turns = id2outputs[(quid, choice_index)]
        if quid not in quid2choices:
            quid2choices[quid] = []
        quid2choices[quid].append({"index": choice_index, "turns": turns})
    return quid2choices


def write_answers(quid2choices, questions, model_id, answer_file):
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    for question in questions:
        question_id = question["question_id"]
        choices = quid2choices[question_id]
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
        "--stop",
        nargs="+",
        default=[],
        help="List of strings that stop the generation when they are generated."
        "The returned output will not contain the stop strings.",
    )
    parser.add_argument(
        "--stop_token_ids",
        nargs="+",
        default=[],
        help="List of tokens that stop the generation when they are"
        "generated. The returned output will contain the stop tokens unless"
        "the stop tokens are sepcial tokens.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Float that controls the randomness of the sampling. Lower"
        "values make the model more deterministic, while higher values make"
        "the model more random. Zero means greedy sampling.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on whether they
                appear in the generated text so far. Values > 0 encourage the llm
                to use new tokens, while values < 0 encourage the llm to repeat
                tokens.""",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on their
                frequency in the generated text so far. Values > 0 encourage the
                llm to use new tokens, while values < 0 encourage the llm to
                repeat tokens.""",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="The number of GPUs to use for distributed"
        "execution with tensor parallelism.",
    )

    args = parser.parse_args()

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
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        stop=args.stop,
        stop_token_ids=args.stop_token_ids,
    )

    reorg_answer_file(answer_file)
