import time
from pathlib import Path

from ime import decode, load_model, load_pinyin_table, train_and_save


ROOT = Path(__file__).parents[1]
MODEL = ROOT / "model.pkl.gz"


def main():
    table = load_pinyin_table(ROOT / "data/拼音汉字表.txt")
    model = load_model(MODEL) if MODEL.exists() else train_and_save(
        ROOT / "corpus/sina_news_gbk",
        ROOT / "data/一二级汉字表.txt",
        MODEL,
    )
    inputs = (ROOT / "data/input.txt").read_text(encoding="utf-8").splitlines()
    answers = (ROOT / "data/answer.txt").read_text(encoding="utf-8").splitlines()
    if len(inputs) != len(answers):
        raise ValueError("输入和答案行数不一致")

    started = time.perf_counter()
    outputs = [decode(line.split(), table, model) for line in inputs]
    elapsed = time.perf_counter() - started
    baseline = [decode(line.split(), table, model, use_starts=False) for line in inputs]
    (ROOT / "data/output.txt").write_text("\n".join(outputs) + "\n", encoding="utf-8")

    total_chars = sum(map(len, answers))
    correct_chars = sum(
        predicted == expected
        for output, answer in zip(outputs, answers)
        for predicted, expected in zip(output, answer)
    )
    correct_sentences = sum(output == answer for output, answer in zip(outputs, answers))
    baseline_chars = sum(
        predicted == expected
        for output, answer in zip(baseline, answers)
        for predicted, expected in zip(output, answer)
    )
    baseline_sentences = sum(output == answer for output, answer in zip(baseline, answers))

    print(f"普通二元：字 {baseline_chars / total_chars:.2%}，句 {baseline_sentences / len(answers):.2%}")
    print(f"字准确率：{correct_chars / total_chars:.2%} ({correct_chars}/{total_chars})")
    print(f"句准确率：{correct_sentences / len(answers):.2%} ({correct_sentences}/{len(answers)})")
    print(f"预测时间：{elapsed:.3f} 秒")
    print(f"模型大小：{MODEL.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
