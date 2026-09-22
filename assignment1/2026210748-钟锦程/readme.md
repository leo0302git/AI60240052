# 拼音输入法

环境：Python 3.13，无第三方依赖。

新浪新闻和两个汉字表使用 GBK 编码，程序用兼容 GBK 的 GB18030 读取；测试输入与程序输出使用 UTF-8。

## 目录

- `corpus/sina_news_gbk/`：新浪新闻 GBK 语料。
- `data/`：拼音汉字表、汉字表和测试数据。
- `main.py`：程序入口。
- `src/ime.py`：训练和二元 Viterbi 解码；句首统计是唯一的改进方法。
- `src/evaluate.py`：测试准确率与速度。
- `src/oj.py`：使用 OJ 提供的 `1_word.txt`、`2_word.txt`，可直接提交。

## 运行

首次运行会从 `corpus/sina_news_gbk/` 训练并生成 `model.pkl.gz`：

```bash
python main.py <data/input.txt >data/output.txt
```

评测：

```bash
python src/evaluate.py
```

模型文件是中间文件，不放入提交压缩包。`data/output.txt` 保留本次实验结果，再次运行时会被覆盖；删除模型后运行即可重新训练。

本机完整实验结果：训练 354,055,241 个汉字约 99 秒，模型 8.86 MiB，501 句预测约 1 秒。普通二元模型字准确率 84.05%、句准确率 39.52%；加入句首统计后分别为 83.90%、40.12%。

## OJ

OJ 会提供 UTF-8 编码的词频表。选择 Python 3，将 `src/oj.py` 上传或粘贴到编辑器即可；本地语料和模型都不需要上传。
