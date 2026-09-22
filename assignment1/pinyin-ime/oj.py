import json
import math
import sys


BIGRAM_WEIGHT = 0.99


def decode(pinyins, unigrams, bigrams):
    first = unigrams[pinyins[0]]
    first_counts = dict(zip(first["words"], first["counts"]))
    first_total = sum(first_counts.values())
    scores = {char: math.log(count / first_total) for char, count in first_counts.items()}
    backtrack = []

    for previous_pinyin, pinyin in zip(pinyins, pinyins[1:]):
        current = unigrams[pinyin]
        current_counts = dict(zip(current["words"], current["counts"]))
        current_total = sum(current_counts.values())
        pairs = bigrams.get(f"{previous_pinyin} {pinyin}", {"words": [], "counts": []})
        pair_counts = {
            pair.replace(" ", ""): count
            for pair, count in zip(pairs["words"], pairs["counts"])
        }
        previous_counts = dict(
            zip(unigrams[previous_pinyin]["words"], unigrams[previous_pinyin]["counts"])
        )

        # 每个当前字都尝试所有前驱，只保留总分最高的一条路径。
        new_scores = {}
        previous_chars = {}
        for char, count in current_counts.items():
            base = count / current_total
            best_previous = None
            best_score = -math.inf
            for previous, previous_score in scores.items():
                conditional = pair_counts.get(previous + char, 0) / previous_counts[previous]
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

    for line in sys.stdin:
        pinyins = line.split()
        print(decode(pinyins, unigrams, bigrams) if pinyins else "")


if __name__ == "__main__":
    main()
