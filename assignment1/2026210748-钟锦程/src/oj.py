import json
import math
import sys


BIGRAM_WEIGHT = 0.99


def decode(pinyins, unigrams, bigrams, unigram_denominator):
    first = unigrams[pinyins[0]]
    scores = {
        char: math.log((count + 1) / unigram_denominator)
        for char, count in zip(first["words"], first["counts"])
    }
    backtrack = []

    for previous_pinyin, pinyin in zip(pinyins, pinyins[1:]):
        current = unigrams[pinyin]
        pairs = bigrams.get(
            f"{previous_pinyin} {pinyin}",
            {"words": [], "counts": []},
        )
        pair_counts = {
            pair.replace(" ", ""): count
            for pair, count in zip(pairs["words"], pairs["counts"])
        }
        previous = unigrams[previous_pinyin]
        previous_counts = dict(zip(previous["words"], previous["counts"]))

        new_scores = {}
        previous_chars = {}
        for char, count in zip(current["words"], current["counts"]):
            # 用全语料的一元概率给没见过的二元组合兜底。
            base = (count + 1) / unigram_denominator # 这次提交检查是不是这里出问题了。这次换用全vocab上的unigram概率作为base而不是同音字集合里的概率，试一下
            best_previous = None
            best_score = -math.inf
            for previous, previous_score in scores.items():
                previous_count = previous_counts[previous]
                conditional = pair_counts.get(previous + char, 0) / previous_count if previous_count else 0
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
    for previous_chars in reversed(backtrack):
        result.append(previous_chars[result[-1]])
    return "".join(reversed(result))


def main():
    with open("1_word.txt", encoding="utf-8") as source:
        unigrams = json.load(source)
    with open("2_word.txt", encoding="utf-8") as source:
        bigrams = json.load(source)

    # 多音字会重复出现，所以先按汉字去重。
    word_counts = {}
    for item in unigrams.values():
        for char, count in zip(item["words"], item["counts"]):
            word_counts[char] = max(word_counts.get(char, 0), count)
    unigram_denominator = sum(word_counts.values()) + len(word_counts)

    for line in sys.stdin:
        pinyins = line.split()
        print(decode(pinyins, unigrams, bigrams, unigram_denominator) if pinyins else "")


if __name__ == "__main__":
    main()
