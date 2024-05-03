"""
Common data structures and utilities.
"""

import ast
import copy
import dataclasses
import glob
import json
import os
import re
import requests
import time
from typing import Optional

import openai
import anthropic
import tiktoken

from fastchat.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
    OPENAI_MODEL_LIST,
    MINIMAX_MODEL_LIST,
    MINIMAX_PRO_MODEL_LIST,
)

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
API_ERROR_OUTPUT_ZH = "您的输入包含敏感信息，请修改后重试"
API_LEN_ERROR_OUTPUT = "$LEN_ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


def get_questions(question):
    qs = []
    if "turns" in question:
        qs = question["turns"]
    elif "questions" in question:
        qs = question["questions"]
    elif "conversation" in question:
        qs = [
            turn
            for turn in question["conversation"]
            if turn["type"] in ("user_query", "bot_intro")
        ]
    else:
        raise ValueError("Failed to get the questions")
    return qs


def get_answers(answer, choice=0):
    return answer["choices"][choice]["turns"]


def get_number_choices(answer):
    return len(answer["choices"])


def get_system(sample, model, system_keys=None):
    if system_keys is not None and isinstance(system_keys, list):
        if ("npc" in model or "longcat" in model) and "character_name" in system_keys and "character" in system_keys:
            system = f"<role_name>\n{sample['character_name']}\n<role_description>\n{sample['character']}"
            for key in system_keys:
                if key not in ("character_name", "character"):
                    if "character_name" in sample[key]:
                        system += f'\n{sample[key].format(character_name=sample["character_name"])}'
                    else:
                        system += f"\n{sample[key]}"

        else:
            system = ""
            for key in system_keys:
                if key in sample:
                    if "character_name" in sample[key]:
                        system += f'\n{sample[key].format(character_name=sample["character_name"])}'
                    else:
                        system += f"\n{sample[key]}"

        return system

    if system_keys is not None and isinstance(system_keys, str) and system_keys in sample:
        system = sample[system_keys].strip()
        return system

    system_insts = [
        "<<SYS>>" + "\n" + sample["system"] + "\n" + "<</SYS>>"
        if "system" in sample
        else "",
        "<<background>>" + "\n" + sample["background"] + "\n" + "<</background>>"
        if "background" in sample
        else "",
        "<<respond_style>>"
        + "\n"
        + sample["respond_style"]
        + "\n"
        + "<</respond_style>>"
        if "respond_style" in sample
        else "",
    ]
    system = "\n".join([s for s in system_insts if s]).strip()
    return system


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename, encoding="utf-8") as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def run_judge_single(question, answer, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    qus = get_questions(question)
    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=qus[0],
            question_2=qus[1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=qus[0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], f"{system_prompt}\n\n{user_prompt}")
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def format_conversation_context(qus, ans, judge, bot_intro_turn=None):
    conversation_context = ""
    if bot_intro_turn is not None:
        if "{turn}" in judge.prompt_template["bot_template"]:
            turn = judge.prompt_template["bot_template"].format(
                turn=0,
                answer=bot_intro_turn["value"],
            )
        else:
            turn = judge.prompt_template["bot_template"].format(
                answer=bot_intro_turn["value"],
            )
        conversation_context += turn

    n_turns = len(qus)
    for i in range(n_turns):
        question = qus[i]
        answer = ans[i]
        if isinstance(qus[i], dict):
            question = qus[i]["value"]
        if "{turn}" in judge.prompt_template["turn_template"]:
            turn = judge.prompt_template["turn_template"].format(
                turn=i+1,
                question=question,
                answer=answer,
            )
        else:
            turn = judge.prompt_template["turn_template"].format(
                question=question,
                answer=answer,
            )
        conversation_context += turn
    return conversation_context


def run_judge_single_more_turns(
    bot_system, qus_list, ans_list, judge, ref_answer, api_dict, max_tokens
):
    """
    ref_answer not supported yet
    """
    kwargs = {}
    model = judge.model_name
    rating = -1

    n_pop = 0
    while len(qus_list) > 0:
        conv = get_conversation_template(model)
        system_prompt = judge.prompt_template["system_prompt"]
        if "{system}" in judge.prompt_template["system_prompt"] and bot_system:
            system_prompt = system_prompt.format(system=bot_system)

        conversation_context = format_conversation_context(
            qus_list, ans_list, judge
        )
        user_prompt = judge.prompt_template["prompt_template"].format(
            conversations=conversation_context
        )

        conv.append_message(conv.roles[0], f"{system_prompt}\n\n{user_prompt}")
        conv.append_message(conv.roles[1], None)

        if model in OPENAI_MODEL_LIST:
            judgment = chat_completion_openai(
                model, conv, temperature=0, max_tokens=max_tokens, top_p=0.5, api_dict=api_dict,
            )
        elif model in ANTHROPIC_MODEL_LIST:
            judgment = chat_completion_anthropic(
                model, conv, temperature=0, max_tokens=2048, api_dict=api_dict,
            )
        else:
            raise ValueError(f"Invalid judge model name: {model}")

        if "consistency" in judge.prompt_template["name"] and len(qus_list) <= 2:
            break

        if judgment == API_LEN_ERROR_OUTPUT:
            n_pop += 1
            print(
                f"Current {len(qus_list)} turns exceed judge's max length, reduce 1 earliest turn."
            )
            qus_list = qus_list[1:]
            ans_list = ans_list[1:]
        else:
            break

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, system_prompt, user_prompt, judgment, n_pop


def play_a_choice_match_single(
        nth_choice: int, match: MatchSingle, output_file: str, system_keys: list,
        api_dict: dict, max_tokens: int, judge_time: int):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )
    question_id = question["question_id"]
    answer_id = answer["answer_id"]
    qus_list = get_questions(question)
    ans_list = get_answers(answer, nth_choice)
    bot_system = get_system(question, match.model, system_keys)
    bot_intro_turn = None
    n_turns = len(qus_list)
    if len(qus_list) - len(ans_list) == 1 and qus_list[0]["type"] in ("bot_intro",):
        print(f"For role-play, first turn can be bot_intro: {qus_list[0]['value']}")
        bot_intro_turn = qus_list[0]
        qus_list = qus_list[1:]
        n_turns = len(qus_list)
    elif len(qus_list) != len(ans_list):
        print(
            f"mis-match length between {len(qus_list)} questions and {len(ans_list)} answers"
        )
        print(f"Reduce both to {min(len(qus_list), len(ans_list))}")
        n_turns = min(len(qus_list), len(ans_list))
        qus_list = qus_list[:n_turns]
        ans_list = ans_list[:n_turns]

    print(f"Judge: evaluating question {question['question_id']} with {n_turns} turns.")
    results = []
    if (
        judge.prompt_template["type"] == "single"
        and "more-turn" in judge.prompt_template["name"]
    ):
        begin_turn_id = 1
        if "granularity" in judge.prompt_template:
            begin_turn_id = judge.prompt_template["granularity"]

        if begin_turn_id == 0:  # session level
            begin_turn_id = len(qus_list)

        n_pop = 0
        for turn in range(begin_turn_id, n_turns + 1):
            _question = qus_list[n_pop:turn]
            _answer = ans_list[n_pop:turn]
            score, system_prompt, user_prompt, judgment, n_pop = run_judge_single_more_turns(
                bot_system, _question, _answer, judge, ref_answer, api_dict, max_tokens,
            )

            result = {
                "question_id": question_id,
                "answer_id": f"{answer_id}_{nth_choice}",
                "category": question["category"],
                "model": model+f"_judge{judge_time}",
                "judge": (judge.model_name, judge.prompt_template["name"]),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "judgment": judgment,
                "score": score,
                "turn": turn,
                "tstamp": time.time(),
                "dimension": judge.prompt_template["dimension"]
                if "dimension" in judge.prompt_template
                else "general",
            }
            print(
                f"question: {question_id}, turn: {turn}, choice: {nth_choice}, model: {model}_judge{judge_time}, "
                f"score: {score}, "
                f"judge: {(judge.model_name, judge.prompt_template['name'])}"
            )
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "a") as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    elif judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "answer_id": f"{answer_id}_{nth_choice}",
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
            "dimension": judge.prompt_template["dimension"]
            if "dimension" in judge.prompt_template
            else "general",
        }
        print(
            f"question: {question_id}, turn: {turn}, choice: {nth_choice}, model: {model}, "
            f"score: {score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)

    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    return results


def play_a_match_single(
        match: MatchSingle, output_file: str, system_keys: list, api_dict: dict, max_tokens: int, judge_time: int):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )
    num_choices = get_number_choices(answer)
    all_choices_results = []
    for nth_choice in range(num_choices):
        results = play_a_choice_match_single(
            nth_choice, match, output_file, system_keys, api_dict, max_tokens, judge_time)
        all_choices_results.extend(results)

    return all_choices_results


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        conv.set_system_message(system_prompt)
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment)
        if not match:
            match = re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return winner, user_prompt, judgment


def play_a_match_pair(match: MatchPair, output_file: str):
    question, model_1, model_2, answer_1, answer_2, judge, ref_answer, multi_turn = (
        match.question,
        match.model_1,
        match.model_2,
        match.answer_1,
        match.answer_2,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "pairwise":
        g1_winner, g1_user_prompt, g1_judgment = run_judge_pair(
            question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
        )
        g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
            question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
        )

        g1_map = {"A": "model_1", "B": "model_2"}
        g2_map = {"A": "model_2", "B": "model_1"}
        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)
        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2

        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g2_user_prompt": g2_user_prompt,
            "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }

        print(
            f"question: {question_id}, turn: {turn}, model_1: {model_1}, model_2: {model_2}, "
            f"g1_winner: {g1_winner}, g2_winner: {g2_winner}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    elif judge.prompt_template["type"] == "single":
        m1_score, m1_user_prompt, m1_judgment = run_judge_single(
            question, answer_1, judge
        )
        m2_score, m2_user_prompt, m2_judgment = run_judge_single(
            question, answer_2, judge
        )

        if abs(m1_score - m2_score) <= TIE_DELTA:
            winner = "tie"
        elif m1_score > m2_score:
            winner = "model_1"
        else:
            winner = "model_2"

        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": winner,
            "g2_winner": winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": m1_user_prompt,
            "g1_judgment": m1_judgment,
            "g2_user_prompt": m2_user_prompt,
            "g2_judgment": m2_judgment,
            "m1_score": m1_score,
            "m2_score": m2_score,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, model_1: {model_1}, model_2: {model_2}, "
            f"winner: {winner}, m1_score: {m1_score}, m2_score: {m2_score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


context_len_exceed = "Please reduce the length"


def chat_completion_openai(model, conv, temperature, max_tokens, top_p=1.0, api_dict=None):
    if api_dict is not None:
        if "api_base" in api_dict:
            openai.api_base = api_dict["api_base"]
        if "api_key" in api_dict:
            openai.api_key = api_dict["api_key"]

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), str(e))
            if len(conv.messages) > 3:  # extra one for system
                conv.reduce_context(2)
                print(
                    f"Try reduce context to fit in the model context length. "
                    f"Remove first two turns, left {conv.get_num_messages()} turns."
                )
            else:
                print("Stop due to length error")
                output = API_LEN_ERROR_OUTPUT
                break
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.error.InvalidRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    output = API_ERROR_OUTPUT
    if not model.startswith("anthropic"):
        model = f"anthropic.{model}"
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            prompt = conv.get_prompt()
            if not prompt.startswith("\n\nHuman:"):
                prompt = "\n\nHuman:" + prompt

            print(f"DEBUG:\n{prompt}")
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_baichuan_NPC(model, conv, bot_setting, bot_name, user_name=None, relation=None, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    response = None
    output = API_ERROR_OUTPUT_ZH
    messages = conv.to_baichuan_api_messages()
    if len(messages) < 4:
        print(f"DEBUG:\nbot_name\n{bot_name}\nbot_setting\n{bot_setting}\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
    else:
        print(f"DEBUG:\n{json.dumps(messages[-1], indent=2, ensure_ascii=False)}")

    for _ in range(API_MAX_RETRY):
        try:
            data = {
                "model": model,  # "Baichuan-NPC-Lite",
                "character_profile": {
                    "character_name": bot_name,
                    "character_info": bot_setting,
                    "user_name": user_name,
                    "user_info": relation
                },
                "messages": messages,
                "temperature": 0.8,
                "top_k": 10,
                "max_tokens": 512,
                "stream": False
            }
            json_data = json.dumps(data)

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
                "security": "true"
            }
            response = requests.post(baichuan_url, data=json_data, headers=headers, timeout=60)
            if response and response.status_code == 450:
                output = f"error code: {response.status_code}, {API_ERROR_OUTPUT_ZH}"
                break
            else:
                output = json.loads(response.text)['choices'][0]['message']['content'].strip()
        except:
            print(f"Baichuan NPC API Error: {response.status_code} -- {response.text}")
            time.sleep(API_RETRY_SLEEP)

    if response and response.status_code == 200:
        output = json.loads(response.text)['choices'][0]['message']['content'].strip()
    else:
        print(f"Baichuan NPC API Error: {response.status_code} -- {response.text}")

    print(json.dumps({"role": "assistant", "content": output}, indent=2, ensure_ascii=False))
    return output


def chat_completion_minimax_pro(
    model, conv, temperature, max_tokens, bot_setting, bot_name, api_dict=None
):
    def is_valid_bot_setting(bot_setting):
        if bot_setting and isinstance(bot_setting, dict):
            return "bot_name" in bot_setting and "content" in bot_setting
        return False

    minimax_api_key = None
    if api_dict is not None and "api_key" in api_dict:
        minimax_api_key = api_dict["api_key"]

    if model == "abab5.5-chat":
        _minimax_api_base = minimax_pro_api_base
    else:
        _minimax_api_base = minimax_api_base

    if temperature == 0:
        temperature = 1e-5  # Minimax temperature必须在(0,1]之间
    default_sender_type = "BOT"  # 指定回复的角色类型，目前只能传BOT
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + minimax_api_key,
    }
    output = API_ERROR_OUTPUT_ZH
    if is_valid_bot_setting(bot_setting):
        bot_setting = bot_setting
    elif isinstance(bot_setting, str) and len(bot_setting) > 0:
        bot_setting = [
            {
                "bot_name": bot_name,
                "content": bot_setting,
            }
        ]
    else:
        raise ValueError(
            f"Invalid minimax bot setting, "
        )

    messages = conv.to_minimax_api_messages()
    if len(messages) < 4:
        print(
            f"DEBUG:\nbot_name\n{bot_name}\nbot_setting\n{bot_setting}\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
    else:
        print(f"DEBUG:\n{json.dumps(messages[-1], indent=2, ensure_ascii=False)}")

    for _ in range(API_MAX_RETRY):
        try:
            payload = {
                "model": model,
                "stream": False,
                "bot_setting": bot_setting,
                "messages": messages,
                "reply_constraints": {
                    "sender_type": default_sender_type,
                    "sender_name": bot_name,
                },
                "tokens_to_generate": max_tokens,
                "temperature": temperature,
            }
            completion = requests.request(
                "POST", _minimax_api_base, headers=headers, json=payload
            )
            if completion is None:
                print(f"Retry due to Error: getting None from requests ")
            elif json.loads(completion.text)["base_resp"]["status_code"] != 0:
                print(f"Retry due to ERROR:", completion.text)
            else:
                completion = requests.request(
                    "POST", _minimax_api_base, headers=headers, json=payload
                )
                output = json.loads(completion.text)["reply"]
                break
        except:
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_minimax(
    model, conv, temperature, max_tokens, bot_setting, bot_name, user_name="我", api_dict=None
):
    minimax_api_key = None
    if api_dict is not None and "api_key" in api_dict:
        minimax_api_key = api_dict["api_key"]

    _minimax_api_base = minimax_api_base

    if temperature == 0:
        temperature = 1e-5  # Minimax temperature必须在(0,1]之间
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + minimax_api_key,
    }
    output = API_ERROR_OUTPUT_ZH
    role_meta = {
       "user_name": user_name,
       "bot_name": bot_name
    }

    messages = conv.to_minimax_api_messages()
    if len(messages) < 4:
        print(f"DEBUG:\nbot_name\n{bot_name}\nbot_setting\n{bot_setting}\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
    else:
        print(f"DEBUG:\n{json.dumps(messages[-1], indent=2, ensure_ascii=False)}")

    for _ in range(API_MAX_RETRY):
        try:
            payload = {
                "model": model,
                "stream": False,
                "prompt": bot_setting,
                "role_meta": role_meta,
                "messages": messages,
                "tokens_to_generate": max_tokens,
                "temperature": temperature,
                "use_standard_sse": False
            }
            completion = requests.request(
                "POST", _minimax_api_base, headers=headers, json=payload
            )
            if completion is None:
                print(f"Retry due to Error: getting None from requests ")
            elif json.loads(completion.text)["base_resp"]["status_code"] != 0:
                print(f"Retry due to ERROR:", completion.text)
            else:
                completion = requests.request(
                    "POST", _minimax_api_base, headers=headers, json=payload
                )
                output = json.loads(completion.text)["reply"].strip()
                break
        except:
            time.sleep(API_RETRY_SLEEP)

    print(json.dumps({"role": "assistant", "content": output}, indent=2, ensure_ascii=False))
    return output


def chat_completion_openai_like(model, conv, api_dict):
    """
    Self-hosted models using openai style api
    """

    default_param = {
        "max_tokens": 512,
        "temperature": 0.87,
        "repetition_penalty": 1.1
    }

    if api_dict is not None and "api_base" in api_dict:
        openai.api_base = api_base = api_dict["api_base"]
    param = model2param.get(model, default_param)
    print(f"Inference with: {json.dumps(param, indent=2)}")
    openai.api_key = "EMPTY"  # Not support yet
    max_tokens = param["max_tokens"]
    output = pruning_context(conv, model, max_tokens, API_ERROR_OUTPUT_ZH)
    if output == API_LEN_ERROR_OUTPUT:
        return output

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            if len(messages) < 4:
                print(f"DEBUG:\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
            else:
                print(f"DEBUG:\n{json.dumps(messages[-1], indent=2, ensure_ascii=False)}")

            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **param
            )
            output = response["choices"][0]["message"]["content"]

            if model in ("yi-34b-v1.8", "yi-34b-v2.10.1"):
                output = output.strip()

            break
        except openai.error.OpenAIError as e:
            print(type(e), str(e))
            if len(conv.messages) > 6:  # extra one for system
                del conv.messages[3:7]
            elif len(conv.messages) > 4:  # extra one for system
                del conv.messages[3:5]
            else:
                print("Stop due to length error")
                output = API_LEN_ERROR_OUTPUT
                break
            time.sleep(API_RETRY_SLEEP)

    print(json.dumps({"role": "assistant", "content": output}, indent=2, ensure_ascii=False))
    return output


def chat_completion_palm(chat_state, model, conv, temperature, max_tokens):
    from fastchat.serve.api_provider import init_palm_chat

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_pairwise_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model_1, model_2 = obj["question_id"], obj["model_1"], obj["model_2"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        if "winner" in obj:
            winner = obj["winner"]
        elif "g1_winner" in obj and "g2_winner" in obj:
            g1_winner, g2_winner = obj["g1_winner"], obj["g2_winner"]
            if g1_winner == g2_winner:
                winner = g1_winner
            else:
                winner = "inconsistent"
        else:
            raise ValueError(f"Invalid keys: {list(obj.keys())}")

        gamekey = (qid, model_1, model_2)
        winners = (winner,)

        judge_dict[judge][gamekey] = {
            "winners": winners,
            "g1_judgment": obj["g1_judgment"],
            "g2_judgment": obj["g2_judgment"],
        }

    # Make the model names sorted in the game keys
    normalized = {}
    for judge, value in judge_dict.items():
        normalized[judge] = normalize_game_key_dict(value)
    return normalized


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "pair-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "pair-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model_1, model_2 = gamekey
        if model_1 < model_2:
            res = judgment_dict[gamekey]
            g1_judgment, g2_judgment = res["g1_judgment"], res["g2_judgment"]
        else:
            new_gamekey = (qid, model_2, model_1)
            res = judgment_dict[new_gamekey]

            model_1, model_2 = model_1, model_2
            g1_judgment, g2_judgment = res["g2_judgment"], res["g1_judgment"]

        return (
            f"**Game 1**. **A**: {model_1}, **B**: {model_2}\n\n"
            f"**Judgment**: {g1_judgment}"
            + f"\n\n`--------------------------`\n\n"
            + f"**Game 2**. **A**: {model_2}, **B**: {model_1}\n\n"
            f"**Judgment**: {g2_judgment}"
        )
    except KeyError:
        return "N/A"


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
