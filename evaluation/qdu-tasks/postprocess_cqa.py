import re
import json
import argparse
from collections import defaultdict

def main(args):
    all_results = defaultdict(list)
    with open(args.tmp_path, encoding="utf-8") as f:
        for line in f:
            result = re.search("Metrics : (\{.*\})", line).group(1)
            result = json.loads(result)
            for k, v in result.items():
                all_results[k].append(v)
    for k, v in all_results.items():
        all_results[k] = sum(v) / len(v)
    print(dict(all_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tmp_path", default="data/results/cqadupstack/tmp.log")
    args = parser.parse_args()

    main(args)
