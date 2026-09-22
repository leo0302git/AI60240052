import sys
from pathlib import Path

from src.ime import decode, load_model, load_pinyin_table, train_and_save


ROOT = Path(__file__).parent
MODEL = ROOT / "model.pkl.gz"


def main():
    table = load_pinyin_table(ROOT / "data/拼音汉字表.txt")
    model = load_model(MODEL) if MODEL.exists() else train_and_save(
        ROOT / "corpus/sina_news_gbk",
        ROOT / "data/一二级汉字表.txt",
        MODEL,
    )

    for line in sys.stdin:
        pinyins = line.split()
        print(decode(pinyins, table, model) if pinyins else "")


if __name__ == "__main__":
    main()
