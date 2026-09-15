# Modified file; the original work is under Apache-2.0, see README (License).
import ast
import os
from pathlib import Path
import pickle
import json
import logging
import datetime
import re
import random
import base64

def get_models(model_name):
    """One adapter for every provider. The provider is inferred from the
    model name (gpt-* -> OpenAI, claude-* -> Anthropic, gemini-*, deepseek-*,
    kimi-*, qwen*, ...) or forced with a "provider:" prefix such as
    "compat:my-local-model"; keys and endpoints come from .env. See
    animask/llm.py for the table."""
    from modules.llm.ChatAPI import ChatAPI
    return ChatAPI(model=model_name)


def build_world_agent_data(world_file_path,max_words = 30):
    world_dir = os.path.dirname(world_file_path)
    details_dir = os.path.join(world_dir,"./world_details")
    data = []
    settings = []
    if os.path.exists(details_dir):
        for path in get_child_paths(details_dir):
            if os.path.splitext(path)[-1] == ".txt":
                text = load_text_file(path)
                data += split_text_by_max_words(text,max_words)
            if os.path.splitext(path)[-1] == ".jsonl":
                jsonl = load_jsonl_file(path)
                data += [f"{dic['term']}:{dic['detail']}" for dic in jsonl]
                settings += jsonl
    return data,settings

def build_db(data, db_name, db_type, embedding, save_type="persistent"):
    if not data or not db_name:
        return None
    # dependency-free lexical retrieval (db_type is kept for call compatibility)
    from modules.db.NaiveDB import NaiveDB
    db = NaiveDB(embedding, save_type)
    db.init_from_data(data, db_name)
    return db

def get_root_dir():
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)
    return root_dir

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(experiment_name):
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_dir(f"{get_root_dir()}/log/{experiment_name}")
    file_handler = logging.FileHandler(os.path.join(get_root_dir(),f"./log/{experiment_name}/{current_time}.log"),encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # Avoid logging duplication
    logger.propagate = False

    return logger

def merge_text_with_limit(text_list, max_words, language = 'en'):
    """
    Merge a list of text strings into one, stopping when adding another text exceeds the maximum count.

    Args:
        text_list (list): List of strings to be merged.
        max_count (int): Maximum number of characters (for Chinese) or words (for English).
        is_chinese (bool): If True, count Chinese characters; if False, count English words.

    Returns:
        str: The merged text, truncated as needed.
    """
    merged_text = ""
    current_count = 0

    for text in text_list:
        if language == 'zh':
            # Count Chinese characters
            text_length = len(text)
        else:
            # Count English words
            text_length = len(text.split(" "))

        if current_count + text_length > max_words:
            break

        merged_text += text + "\n"
        current_count += text_length

    return merged_text

def normalize_string(text):
    # 去除空格并将所有字母转为小写
    import re
    return re.sub(r'[\s\,\;\t\n]+', '', text).lower()

def fuzzy_match(str1, str2, threshold=0.8):
    str1_normalized = normalize_string(str1)
    str2_normalized = normalize_string(str2)

    if str1_normalized == str2_normalized:
        return True

    return False

def remove_list_elements(list1, *args):
    for target in args:
        if isinstance(target,list) or isinstance(target,dict):
            list1 = [i for i in list1 if i not in target]
        else:
            list1 = [i for i in list1 if i != target]
    return list1

def load_text_file(path):
    with open(path,"r",encoding="utf-8") as f:
        text = f.read()
    return text

def save_text_file(path,target):
    with open(path,"w",encoding="utf-8") as f:
        text = f.write(target)

def load_json_file(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)
    
def save_json_file(path,target):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False,indent=True)
        
def load_jsonl_file(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
        
def save_jsonl_file(path,target):
    with open(path, "w",encoding="utf-8") as f:
        for row in target:
            print(json.dumps(row, ensure_ascii=False), file=f)

def split_text_by_max_words(text: str, max_words: int = 30):
    segments = []
    current_segment = []
    current_length = 0
    
    lines = text.splitlines()

    for line in lines:
        words_in_line = len(line)
        current_segment.append(line + '\n')
        current_length += words_in_line
        
        if current_length + words_in_line > max_words:
            segments.append(''.join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(''.join(current_segment))

    return segments

def lang_detect(text):
    import re
    def count_chinese_characters(text):
        # 使用正则表达式匹配所有汉字字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars)
            
    if count_chinese_characters(text) > len(text) * 0.05:
        lang = 'zh'
    else:
        lang = 'en'
    return lang

def dict_to_str(dic):
    res = ""
    for key in dic:
        res += f"{key}: {dic[key]};"
    return res

def json_parser(output):
    output = output.replace("\n", "")
    output = output.replace("\t", "")
    if "{" not in output:
        output = "{" + output
    if "}" not in output:
        output += "}"  
    pattern = r'\{.*\}'
    matches = re.findall(pattern, output, re.DOTALL)
    try:
        parsed_json = json.loads(matches[0])
    except json.JSONDecodeError:
        try:
            # ast.literal_eval, NOT eval: same tolerance for Python-style
            # dicts (single quotes, True/None) without executing arbitrary
            # expressions from model output
            parsed_json = ast.literal_eval(matches[0])
        except (ValueError, SyntaxError):
            try:
                detail = re.search(r'"detail":\s*(.+?)\s*}', matches[0]).group(1)
                detail = f"\"{detail}\"" 
                new_output = re.sub(r'"detail":\s*(.+?)\s*}', f"\"detail\":{detail}}}", matches[0])
                parsed_json = json.loads(new_output)
            except Exception as e:
                raise ValueError("No valid JSON found in the input string")
    return parsed_json

def action_detail_decomposer(detail):
    thoughts = re.findall(r'【(.*?)】', detail)
    actions = re.findall(r'（(.*?)）', detail)
    dialogues = re.findall(r'「(.*?)」', detail)
    return thoughts,actions,dialogues

def conceal_thoughts(detail):
    # re.S: inner monologue wrapped across lines (【…\n…】) must not leak
    text = re.sub(r'【.*?】', '', detail, flags=re.S)
    text = re.sub(r'\[.*?\]', '', text, flags=re.S)
    return text


def as_bool(value):
    """LLM JSON booleans arrive as true / "true" / "false" — a non-empty
    "false" string must never count as true."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def location_alias_key(name):
    """Normalize a location name for alias matching. Keeps CJK (the old
    ASCII-only rule mapped every zh name to "", so all zh locations
    collapsed onto one alias key)."""
    return re.sub(r"[^A-Za-z0-9一-鿿]", "", str(name or "")).lower()


def resolve_location_alias(destination, locations_info):
    """Resolve a freely-named destination against known location codes and
    names. Returns a location_code, or None when nothing matches (the
    caller may then register the place as new). Empty keys never match."""
    key = location_alias_key(destination)
    if not key:
        return None
    for code in locations_info:
        if location_alias_key(code) == key:
            return code
    byname = {}
    for code, info in locations_info.items():
        k = location_alias_key(info.get("location_name", ""))
        if k:
            byname.setdefault(k, code)
    return byname.get(key)


def pick_location_by_name(wanted, locations_info, default_code):
    """Choose the location whose name best matches a free-text position.
    The LONGEST contained name wins (first containment used to win, so
    "皇宫内侍局宿舍门口" landed on 皇宫 instead of 内侍局宿舍); otherwise
    require >=2 overlapping tokens, else the default."""
    wanted = (wanted or "").lower()

    def _toks(s):
        return set(re.findall(r"[a-z]+|[一-鿿]+", s.lower())) \
            | set(re.findall(r"[一-鿿]", s))

    best_code, best_len = None, 0
    for code, li in locations_info.items():
        name = (li.get("location_name") or "").lower()
        if name and name in wanted and len(name) > best_len:
            best_code, best_len = code, len(name)
    if best_code is not None:
        return best_code
    want_toks = _toks(wanted)
    chosen, best = default_code, 0
    for code, li in locations_info.items():
        overlap = len(_toks(li.get("location_name") or "") & want_toks)
        if overlap > best and overlap >= 2:
            chosen, best = code, overlap
    return chosen


# P0 arm: the card-material placeholder. MUST stay in sync with
# animask/persona_probe/l3_ablation.PLACEHOLDER — the single-step and
# whole-run ablations measure the same construct only if they share it.
P0_PLACEHOLDER = {"zh": "你是这个故事中的一名角色。",
                  "en": "You are a character in this story."}


def seed_from_tag(tag):
    """Replicate number from a run tag ("batch2_r3" -> 3, "r1" -> 1),
    None when the tag carries no rN suffix. A label, not a sampling seed."""
    m = re.search(r"_r(\d+)$|^r(\d+)$", str(tag or ""))
    return int(next(g for g in m.groups() if g)) if m else None


def norm_event_key(name):
    """Correlation key for LLM-echoed event/fixture names. The auditor may
    answer "Winter School Break" for fixture winter_school_break; exact
    string matching then treats every verdict as missing — the audit
    re-fires each run, the cache is never written, and all fixtures are
    silently dropped."""
    return re.sub(r"[^a-z0-9一-鿿]+", "_", str(name or "").lower()).strip("_")


def decode_location_codes(txt, locations_info):
    """Render every registered location code as its display name. Longest
    code first — pinyin codes are heavily prefix-nested, and replacing
    LianShiJiTuan before LianShiJiTuanDaSha leaves hybrid text like
    「廉氏集团DaSha」 in the narrative."""
    if not isinstance(txt, str):
        return txt
    for code, li in sorted(locations_info.items(),
                           key=lambda kv: len(kv[0]), reverse=True):
        name = li.get("location_name")
        if name and name != code and code in txt:
            txt = txt.replace(code, name)
    return txt


def strip_wrapping_quotes(text):
    # world-LLM outputs (event summaries, settlements) sometimes
    # arrive wrapped in quotes; strip only symmetric outer pairs
    if not isinstance(text, str):
        return text
    t = text.strip()
    pairs = [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'), ('「', '」')]
    changed = True
    while changed and len(t) >= 2:
        changed = False
        for a, b in pairs:
            if t.startswith(a) and t.endswith(b):
                t = t[1:-1].strip()
                changed = True
    return t

def extract_first_number(text):
    match = re.search(r'\b\d+(?:\.\d+)?\b', text)
    return int(match.group()) if match else None

def check_role_code_availability(role_code,role_file_dir):
    for path in get_grandchild_folders(role_file_dir):
        if role_code in path:
            return True
    return False
    
def get_grandchild_folders(root_folder, if_full = True):
    folders = []
    for resource in os.listdir(root_folder):
        subpath = os.path.join(root_folder,resource)
        for folder_name in os.listdir(subpath):
            folder_path = os.path.join(subpath, folder_name)
            if if_full:
                folders.append(folder_path)
            else:
                folders.append(folder_name)
    
    return folders

def get_child_folders(root_folder, if_full = True):
    folders = []
    for resource in os.listdir(root_folder):
        if if_full:
            path = os.path.join(root_folder,resource)
            if os.path.isdir(path):
                folders.append(path)
        else:
            path = resource
            if os.path.isdir(os.path.join(root_folder, path)):
                folders.append(path)
    return folders

def get_child_paths(root_folder, if_full = True):
    paths = []
    for resource in os.listdir(root_folder):
        if if_full:
            path = os.path.join(root_folder,resource)
            if os.path.isfile(path):
                paths.append(path)
        else:
            path = resource
            if os.path.isfile(os.path.join(root_folder, path)):
                paths.append(path)
    return paths

def get_first_directory(path):
    try:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                return full_path
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def find_files_with_suffix(directory, suffix):
    matched_files = []
    for root, dirs, files in os.walk(directory):  # 遍历目录及其子目录
        for file in files:
            if file.endswith(suffix):  # 检查文件后缀
                matched_files.append(os.path.join(root, file))  # 将符合条件的文件路径加入列表

    return matched_files

def remove_element_with_probability(lst, threshold=3, probability=0.2):
    # 确保列表不为空
    if len(lst) > threshold and random.random() < probability:
        # 随机选择一个元素的索引
        index = random.randint(0, len(lst) - 1)
        # 删除该索引位置的元素
        lst.pop(index)
    return lst
  
def clean_collection_name(name: str) -> str:
    cleaned_name = name.replace(' ', '_')
    cleaned_name = cleaned_name.replace('.', '_')
    if not all(ord(c) < 128 for c in cleaned_name):
        encoded = base64.b64encode(cleaned_name.encode('utf-8')).decode('ascii')
        encoded = encoded[:60] if len(encoded) > 60 else encoded
        valid_name = f"mem_{encoded}"
    else:
        valid_name = cleaned_name
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '-', valid_name)
    valid_name = re.sub(r'\.\.+', '-', valid_name)
    valid_name = re.sub(r'^[^a-zA-Z0-9]+', '', valid_name)  # 移除开头非法字符
    valid_name = re.sub(r'[^a-zA-Z0-9]+$', '', valid_name)
    valid_name = valid_name[:60]
    return valid_name
    
cache_sign = True
cache = None 
def cached(func):
    def wrapper(*args,**kwargs):
        global cache
        cache_path = "bw_cache.pkl"
        if cache == None:
            if not os.path.exists(cache_path):
                cache = {}
            else:
                cache = pickle.load(open(cache_path, 'rb'))
        key = (func.__name__, str([args[0].role_code, args[0].__class__, args[0].llm_name , args[0].history]), str(kwargs.items()))
        if (cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']) :
            return cache[key]
        else:
            result = func(*args, **kwargs)
            if result != 'busy' and result != None:
                cache[key] = result
                pickle.dump(cache, open(cache_path, 'wb'))
            return result
    return wrapper
