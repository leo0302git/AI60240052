from oj import decode


unigrams = {
    "ren": {"words": ["人", "仁"], "counts": [10, 1]},
    "gong": {"words": ["工", "公"], "counts": [8, 4]},
}
bigrams = {
    "ren gong": {"words": ["人工", "仁公"], "counts": [7, 1]},
}

assert decode(["ren", "gong"], unigrams, bigrams, 27) == "人工"
print("OJ decoder self-check passed")
