import re
import sys
import string
import json
from typing import Union, List, Tuple, Dict, Any
from collections import Counter

import numpy as np
# from src.verl.utils.reward_score.if_eval.instruction_following_algorithm_environment import IF_FUNCTIONS_MAP
from .if_eval.instruction_following_algorithm_environment import IF_FUNCTIONS_MAP, need_code_functions


def validate_format(text: str, use_tool: str = "general") -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # check the order of tool/result: tool invocation and tool response results
    def check_search_format(text: str) -> tuple[bool, str]:
        """
        Checks if <search> and <result> tags are correctly formatted and ordered within the text.
        Handles multiple occurrences of tool/result blocks.
        """
        current_pos = 0
        has_search_tags = False

        if text.find('<code>', current_pos) != -1:
            return False, "Incorrect tool usage."

        while True:
            search_start_pos = text.find('<search>', current_pos)

            if search_start_pos == -1:
                # No more <search> tags found
                break

            has_search_tags = True
            search_end_pos = text.find('</search>', search_start_pos)
            result_start_pos = text.find('<result>', search_start_pos)
            result_end_pos = text.find('</result>', result_start_pos)

            if -1 in (search_end_pos, result_start_pos, result_end_pos):
                return False, "Incomplete <search>/<result> tags."

            if not (search_start_pos < search_end_pos < result_start_pos < result_end_pos):
                return False, "Incorrect order or nesting of <search>/<result> tags."

            current_pos = result_end_pos

        if not has_search_tags:
            return False, "No <search> tags found."

        return True, "<search>/<result> tags are correctly formatted."

    def check_code_format(text: str) -> tuple[bool, str]:
        """
        Checks if <code> and <result> tags are correctly formatted and ordered within the text.
        Handles multiple occurrences of tool/result blocks.
        """
        current_pos = 0
        has_code_tags = False

        if text.find('<search>', current_pos) != -1:
            return False, "Incorrect tool usage."

        while True:
            code_start_pos = text.find('<code>', current_pos)

            if code_start_pos == -1:
                # No more <code> tags found
                break

            has_code_tags = True
            code_end_pos = text.find('</code>', code_start_pos)
            result_start_pos = text.find('<result>', code_start_pos)
            result_end_pos = text.find('</result>', result_start_pos)

            if -1 in (code_end_pos, result_start_pos, result_end_pos):
                return False, "Incomplete <code>/<result> tags."

            if not (code_start_pos < code_end_pos < result_start_pos < result_end_pos):
                return False, "Incorrect order or nesting of <code>/<result> tags."

            current_pos = result_end_pos

        if not has_code_tags:
            return False, "No <code> tags found."

        return True, "<code>/<result> tags are correctly formatted."

    def check_nontool_format(text: str) -> tuple[bool, str]:
        if text.count('<search>') != 0 or text.count('</search>') != 0:
            return False, "Non-ToRL response contains <search> or </search> tags"
        if text.count('<code>') != 0 or text.count('</code>') != 0:
            return False, "Non-ToRL response contains <code> or </code> tags"
        if text.count('<result>') != 0 or text.count('</result>') != 0:
            return False, "Non-ToRL response contains <result> or </result> tags"

        return True, "Non-ToRL format is correct"

    # check if response text contains tool invocation format
    if use_tool == "code":
        flag, reason = check_code_format(text)
    elif use_tool == "search":
        flag, reason = check_search_format(text)
    else:
        flag, reason = check_nontool_format(text)  # not use_tool

    if not flag:
        return False, reason

    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        # if normalized_ground_truth in ["a", "b", "c", "d"] and normalized_prediction != normalized_ground_truth:
        #     continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']


def get_ifeval_score(answer: str, ground_truth: Union[str, List[str], np.ndarray]) -> Tuple[float, str]:
    """
    1) 从 ground_truth 中解析出 func_name 和 args, RLVR-IFeval 仅存在一个唯一的func_name: verifier
    2) 在 IF_FUNCTIONS_MAP 中查找对应 verifier
    3) 用 answer 调用它，返回 (if_score, 消息)
    """
    # print(type(ground_truth), ground_truth)

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]  # ["{}"],获取第一个json.dumps后的str(dict)元素
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth[0]  # 提取第一个元素

    try:
        parsed = json.loads(ground_truth)  # input type is str
    except json.JSONDecodeError as e:
        return 0.0, f"invalid ground_truth JSON: {e}"

    func_name = parsed.get("func_name")
    if not func_name:
        return 0.0, "ground_truth missing func_name"

    args = {k: v for k, v in parsed.items() if k != "func_name" and v is not None}

    verifier = IF_FUNCTIONS_MAP.get(func_name)
    if verifier is None:
        return 0.0, f"verifier '{func_name}' not found"

    try:
        raw = verifier(answer, **args)
        if isinstance(raw, tuple):
            score = float(raw[0])
        elif isinstance(raw, bool):
            score = float(raw)
        else:
            score = 0.0
        # print(score, f"verifier returned {raw}")
        return score, f"verifier returned {raw}"
    except Exception as e:
        return 0.0, f"verifier exception: {e}"


def compute_score(tokenizer, solution_str, ground_truth, use_tool="general") -> Tuple[float, str]:
    # handling both the base model and the instruction-tuned model
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")

    response = solution_str_split[1]

    # special deal for RLVR-IFeval benchmark
    if "func_name" in ground_truth[0]:
        try:
            parsed = json.loads(ground_truth[0])  # input type is str
        except json.JSONDecodeError as e:
            return 0.0, f"invalid ground_truth JSON: {e}"

        func_name = parsed.get("func_name")
        if func_name in need_code_functions.keys():
            use_tool = "code"
        else:
            use_tool = "general"

    valid_template, reason = validate_format(response, use_tool)

    if not valid_template:
        # Incorrect tool utilization will be punished
        if reason == "Incorrect tool usage." or "Non-ToRL response contains" in reason:
            return -0.1, f'bad format due to {reason}'  # Punishment: The reward value is -0.1
        else:
            return 0, f'bad format: {reason}'

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0, f'over length'

    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            return 0, f'find box error: {e}'
    else:
        return 0, f'cannot extract answer'

    # compute IFEval score
    if "func_name" in ground_truth[0]:
        if_score, if_info = get_ifeval_score(answer, ground_truth)
        if if_score > 0:
            return if_score, f'correct answer, get ifeval score: {if_score}'
        else:
            return 0.1, f'wrong answer but good format: {answer}'
    else:
        # KL Divergence: Instruction Alignment
        f1_score = get_f1_score(answer, ground_truth)
        if f1_score > 0:
            return f1_score, f'correct answer, get f1 score: {f1_score}'
        else:
            return 0.1, f'wrong answer but good format: {answer}'

# if __name__ == '__main__':
#     answer = "Raymond III was the Count of Tripoli from 1152 to 1187 and Prince of Galilee and Tiberias in the Kingdom of Jerusalem. He was born in 1140 to Raymond II of Tripoli and Hodierna of Jerusalem. His mother was the daughter of Baldwin II of Jerusalem. \n\n*Early Life and Succession*\n\nRaymond III was only a child when his father was murdered. His mother Hodierna was regent until Raymond came of age. In 1155 Raymond married Eschiva the daughter of Walter I of Beirut. They had three children: Raymond IV Bertrand and a daughter who married Guy of Lusignan. \n\n*Reign*\n\nRaymond III's reign was marked by a series of conflicts and alliances with the Muslim world. He was involved in the defense of the Kingdom of Jerusalem against Nur ad-Din and later Saladin. He was also a key figure in the internal politics of the kingdom. He was a supporter of the queen mother Amalric of Jerusalem and opposed the succession of Guy of Lusignan. \n\n*Later Life and Death*\n\nIn 1187 Raymond III was part of the disastrous Battle of Hattin where the Christian forces were defeated by Saladin. He was one of the few to escape the battlefield but died later that year. His son Raymond IV succeeded him as Count of Tripoli. \n\nRaymond III's life and reign were marked by the complex politics and warfare of the Crusader states. He was a key figure in the defense of the Kingdom of Jerusalem and his death marked a significant loss for the Christian forces in the region. His legacy is a testament to the turbulent times in which he lived and the challenges faced by the Crusader states in their relations with the Muslim world."
#     import pandas as pd
#     test_df = pd.read_parquet("/share/project/weiyifan/ReSearch/data/RLVR-IFeval/test_rlvr_ifeval.parquet")
#     test_df['score'] = test_df.apply(lambda row: get_ifeval_score(answer, row['reward_model']['ground_truth']), axis=1)
