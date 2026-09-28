import gzip
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


BIGRAM_WEIGHT = 0.99


def read_gbk(path):
    """附件是 GBK；GB18030 向下兼容 GBK，也能覆盖更多汉字。"""
    return path.read_text(encoding="gb18030")


def load_pinyin_table(path):
    table = {}
    for line in read_gbk(path).splitlines():
        pinyin, *chars = line.split() # 因为拼音汉字表的形式是：拼音 汉字1 汉字2 ...所以可以这样拆包
        table[pinyin] = chars
    return table # 返回格式类似： table = {"ren": ["人", "仁", "壬", "忍"], "gong": ["工", "公", "功", "攻"], ...}


def train(corpus_dir, charset_path):
    allowed = set(read_gbk(charset_path).strip()) # 只处理一二级汉字表中的字
    unigram = Counter() # 统计一个字的出现次数
    bigram = defaultdict(Counter) # 统计两个字相邻出现的次数，格式类似：bigram = {"人": Counter({"工": 7, "公": 3}), "仁": Counter({"公": 1})}。相当于记录了条件概率
    starts = Counter() # 统计句子开头的字，格式类似：starts = {"人": 10, "仁": 5}
    total = 0

    started = time.perf_counter()
    files = sorted(corpus_dir.glob("2016-*.txt"))
    if not files:
        raise FileNotFoundError(f"未找到训练语料：{corpus_dir}")

    for path in files:
        print(f"训练 {path.name}", file=sys.stderr)
        with path.open(encoding="gb18030") as source:
            for line in source:
                try:
                    news = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for text in (news.get("title", ""), news.get("html", "")):
                    previous = None
                    for char in text:
                        if char not in allowed:
                            # 标点会隔开句子，不能把两边的字算成相邻。
                            previous = None # 重置previous
                            continue
                        if previous is None:
                            starts[char] += 1
                        unigram[char] += 1
                        total += 1
                        if previous:
                            bigram[previous][char] += 1
                        previous = char

    elapsed = time.perf_counter() - started
    print(f"训练完成：{total:,} 字，{elapsed:.2f} 秒", file=sys.stderr)
    return {
        "unigram": unigram,
        "bigram": dict(bigram),
        "starts": starts,
        "total": total,
    }


def train_and_save(corpus_dir, charset_path, model_path):
    model = train(corpus_dir, charset_path)
    with gzip.open(model_path, "wb") as target:
        pickle.dump(model, target)
    print(f"模型已保存：{model_path}", file=sys.stderr)
    return model


def load_model(path):
    with gzip.open(path, "rb") as source:
        return pickle.load(source)


def decode(pinyins, table, model, use_starts=True):
    if not pinyins:
        return ""

    unigram = model["unigram"]
    bigram = model["bigram"]
    starts = model["starts"]
    total = model["total"]
    vocabulary = len(unigram)

    candidates = []
    for pinyin in pinyins:
        if pinyin not in table:
            raise ValueError(f"未知拼音：{pinyin}")
        candidates.append(table[pinyin]) # 得到一个list，里面是每个拼音对应的所有候选汉字，类似：candidates = [["人", "仁"], ["工", "公"],...]

    def unigram_probability(char):
        # 加一保证没在语料里出现过的候选字也有非零概率。
        return (unigram[char] + 1) / (total + vocabulary)

    start_total = sum(starts.values())
    # 用“句首字频率”作为初始分数，更符合真实语言习惯。
    if use_starts:
        scores = {
            char: math.log((starts[char] + 1) / (start_total + vocabulary))
            for char in candidates[0] # 这里已经得到了第一个拼音对应的所有候选字，然后计算每个候选字作为句首的概率，并取对数作为初始分数
        }
    else: # 不用句首字频率的话，就退化为用普通的 unigram 频率作为初始分数
        scores = {char: math.log(unigram_probability(char)) for char in candidates[0]}
    backtrack = []
    # 这里得到了一个score字典：{char: score}

    for chars in candidates[1:]: # 所以这里从第二个拼音对应的所有候选字开始
        new_scores = {}
        previous_chars = {}
        for char in chars: # 对每个候选字，计算它的最优前驱字和对应的分数
            base = unigram_probability(char)
            best_previous = None
            best_score = -math.inf
            for previous, previous_score in scores.items():
                pair = bigram.get(previous, {}).get(char, 0) # bigram 类似 {"人": Counter({"工": 7, "公": 3}), "仁": Counter({"公": 1})}
                # 这里得到了对于某一个候选字char，与他的某个前驱字previous的bigram出现次数，记录在pair中，即 p(char|previous)
                conditional = pair / unigram[previous] if unigram[previous] else 0 # 计算条件概率 p(char|previous) = count(previous, char) / count(previous)
                probability = BIGRAM_WEIGHT * conditional + (1 - BIGRAM_WEIGHT) * base # 用unigram平滑一下，是使得没有出现过的pair也有非零概率
                candidate_score = previous_score + math.log(probability)
                if candidate_score > best_score:
                    best_previous = previous
                    best_score = candidate_score

            new_scores[char] = best_score
            previous_chars[char] = best_previous
            # 这里表示：char的最优前驱字是best_previous，对应的分数是best_score

        scores = new_scores # 记录了这个拼音对应的所有候选字的最优分数
        backtrack.append(previous_chars) # 记录了这个拼音对应的所有候选字的最优前驱字
        # new_scores = {"工": -3.562, "公": -4.1}
        # previous_chars = {"工": "人", "公": "人"}

    # 跑出循环后，scores里面存着最后一个拼音对应的所有候选字的最优分数，backtrack里面（按时间顺序）存着每个拼音对应的所有候选字的最优前驱字
    result = [max(scores, key=scores.get)] # 取出整句的最后一个字
    # 从最后一个字沿着记录的最优前驱倒推整句。
    for previous_chars in reversed(backtrack):
        result.append(previous_chars[result[-1]]) # 挨着倒推，result[-1]是当前的字，previous_chars[result[-1]]就是它的最优前驱字，append到result中
    return "".join(reversed(result))


def _self_check():
    table = {"ren": ["人", "仁"], "gong": ["工", "公"]}
    model = {
        "unigram": Counter({"人": 10, "仁": 1, "工": 8, "公": 4}),
        "bigram": {"人": Counter({"工": 7}), "仁": Counter({"公": 1})},
        "starts": Counter({"人": 8, "仁": 1}),
        "total": 23,
    }
    assert decode(["ren", "gong"], table, model) == "人工"
    print("self-check passed")


if __name__ == "__main__":
    _self_check()
