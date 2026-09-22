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
        pinyin, *chars = line.split()
        table[pinyin] = chars
    return table


def train(corpus_dir, charset_path):
    allowed = set(read_gbk(charset_path).strip())
    unigram = Counter()
    bigram = defaultdict(Counter)
    starts = Counter()
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
                            previous = None
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
        candidates.append(table[pinyin])

    def unigram_probability(char):
        # 加一保证没在语料里出现过的候选字也有非零概率。
        return (unigram[char] + 1) / (total + vocabulary)

    start_total = sum(starts.values())
    if use_starts:
        scores = {
            char: math.log((starts[char] + 1) / (start_total + vocabulary))
            for char in candidates[0]
        }
    else:
        scores = {char: math.log(unigram_probability(char)) for char in candidates[0]}
    backtrack = []

    for chars in candidates[1:]:
        new_scores = {}
        previous_chars = {}
        for char in chars:
            base = unigram_probability(char)
            best_previous = None
            best_score = -math.inf
            for previous, previous_score in scores.items():
                pair = bigram.get(previous, {}).get(char, 0)
                conditional = pair / unigram[previous] if unigram[previous] else 0
                probability = BIGRAM_WEIGHT * conditional + (1 - BIGRAM_WEIGHT) * base
                candidate_score = previous_score + math.log(probability)
                if candidate_score > best_score:
                    best_previous = previous
                    best_score = candidate_score

            new_scores[char] = best_score
            previous_chars[char] = best_previous

        scores = new_scores
        backtrack.append(previous_chars)

    result = [max(scores, key=scores.get)]
    # 从最后一个字沿着记录的最优前驱倒推整句。
    for previous_chars in reversed(backtrack):
        result.append(previous_chars[result[-1]])
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
