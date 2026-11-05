import sys
def main():
    for line in sys.stdin: # 读题，每一行是一个拼音序列，空格分隔
        pinyins = line.split()
        print(pinyins)


if __name__ == "__main__":
    main()