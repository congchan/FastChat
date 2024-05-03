"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
    API_LEN_ERROR_OUTPUT,
    chat_completion_anthropic,
    chat_completion_palm,
    chat_completion_minimax_pro,
    get_questions,
    get_system,
    chat_completion_baichuan_NPC,
    chat_completion_minimax,
    chat_completion_openai_like,
)

from fastchat.llm_judge.gen_model_answer import (
    reorg_answer_file,
    category_with_role,
)

from fastchat.model.model_adapter import (
    get_conversation_template,
    OPENAI_MODEL_LIST,
    ANTHROPIC_MODEL_LIST,
    MINIMAX_MODEL_LIST,
    MINIMAX_PRO_MODEL_LIST,
    BAICHUAN_MODEL_LIST, 
    BAICHUAN_NPC_MODEL_LIST, 
    DOCKER_MODEL_LIST,
)


def get_bot_setting_from_data(sample):
    """
    Minimax parameters：
        bot_setting 对每一个机器人的设定
        bot_setting.bot_name 具体机器人的名字
        bot_setting.content 具体机器人的设定

    Baichuan parameters:
        'character_name': 具体机器人的名字,
        'character_info': 具体机器人的设定,

    return
        bot_name, content (or bot_setting)
    """
    bot_setting = None
    bot_name = "assistant"
    content = None
    if "bot_setting" in sample:
        bot_setting = sample["bot_setting"]
    elif "character" in sample:
        bot_setting = sample["character"]
    elif "background" in sample:
        content = sample["background"]
    elif "respond_style" in sample:
        content = sample["respond_style"]
    elif "system" in sample:
        content = sample["system"]

    if "assistant_role" in sample:
        bot_name = sample["assistant_role"]
    elif "assistant_name" in sample:
        bot_name = sample["assistant_name"]
    elif "character_name" in sample:
        bot_name = sample["character_name"]

    if content:
        return bot_name, content

    return bot_name, bot_setting


def get_answer(
    question: dict, system_key: str, model: str, api_dict: dict, num_choices: int, max_tokens: int, answer_file: str
):
    def _is_possible_chat_with_roles():
        if question["category"].lower() in category_with_role:
            return True
        if isinstance(qs, dict) and "from" in qs:
            return True

        return False

    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    if_exceed_api_length = False
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        qss = get_questions(question)
        system_p = get_system(question, model, system_key)
        if system_p:
            conv.set_system_message(system_p)

        user_role = conv.roles[0]
        user_value = None
        if_infer_greeting = False
        if (
            len(qss) > 1
            and isinstance(qss[0], dict)
            and qss[0]["type"] in ("bot_intro", "greeting")
        ):
            bot_intro_turn = qss[0]

            if len(bot_intro_turn["value"]) == 0:
                print("Need to generate greeting.")
                if_infer_greeting = True
            else:
                conv.append_message_with_type(
                    conv.roles[1], bot_intro_turn["value"], "bot_query"
                )

            qss = qss[1:]

        for j in range(len(qss)):
            if j == 0 and if_infer_greeting:
                assistant_role = conv.roles[1]
                conv.append_message_with_type(user_role, "", "user_query")
                conv.append_message_with_type(assistant_role, None, "bot_query")
                if_infer_greeting = False
            else:
                qs = qss[j]
                if _is_possible_chat_with_roles():
                    user_role = qs["from"]
                    user_value = qs["value"]
                    assistant_role = question.get("assistant_role", conv.roles[1])
                else:
                    user_role = conv.roles[0]
                    user_value = qs
                    assistant_role = conv.roles[1]

                if model in ANTHROPIC_MODEL_LIST:
                    assistant_role = conv.roles[1]

                conv.append_message_with_type(user_role, user_value, "user_query")
                conv.append_message_with_type(assistant_role, None, "bot_query")

            if if_exceed_api_length:
                output = API_LEN_ERROR_OUTPUT
            elif model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(
                    model, conv, temperature, max_tokens, api_dict,
                )
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif model in MINIMAX_PRO_MODEL_LIST or model in MINIMAX_MODEL_LIST:
                bot_name, bot_setting = get_bot_setting_from_data(question)
                output = chat_completion_minimax(
                    model, conv, temperature, max_tokens, bot_setting, bot_name, api_dict=api_dict
                )
            elif model in BAICHUAN_NPC_MODEL_LIST:
                bot_name, bot_setting = get_bot_setting_from_data(question)
                output = chat_completion_baichuan_NPC(
                    model, conv, bot_setting, bot_name, user_name="陌生人", relation="陌生人", api_dict=api_dict
                )
            elif model in OPENAI_MODEL_LIST:
                output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
            else:
                output = chat_completion_openai_like(model, conv, api_dict)

            if output == API_LEN_ERROR_OUTPUT:
                if_exceed_api_length = True

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "category": question["category"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "system_p": system_p,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--system-key",
        type=str,
        nargs="+",
        default=None,
        help="The key to extract system prompt."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_base is not None:
        openai.api_base = args.openai_api_base
    if args.api_key is not None:
        openai.api_key = args.api_key

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")
    api_dict = {}
    if args.api_key is not None:
        api_dict["api_key"] = args.api_key
    if args.api_base is not None:
        api_dict["api_base"] = args.api_base

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.system_key,
                args.model,
                api_dict,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
