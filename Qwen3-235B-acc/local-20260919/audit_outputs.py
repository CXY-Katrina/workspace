"""Audit GSM8K eval details without mistaking repeated numeric answers for duplication."""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

NGRAM_WORDS = 16
MIN_NGRAM_REPEATS = 8
MIN_DUPLICATE_CHARS = 80


def inspect_text(text):
    flags = {}
    if not text.strip():
        flags["empty_output"] = True
    replacement = text.count("\ufffd")
    if replacement:
        flags["replacement_characters"] = replacement
    controls = Counter(
        f"U+{ord(c):04X}" for c in text
        if unicodedata.category(c) in {"Cc", "Cs", "Co", "Cn"} and c not in "\n\r\t"
    )
    if controls:
        flags["unusual_unicode"] = dict(controls)
    mojibake = sum(text.count(marker) for marker in ("Ã", "Â", "â€", "ðŸ", "锟斤拷"))
    if mojibake >= 3:
        flags["possible_mojibake"] = mojibake
    repeated_chars = re.search(r"([^\s])\1{49,}", text)
    if repeated_chars:
        flags["long_character_run"] = repeated_chars.group()[:160]
    words = re.findall(r"\S+", text)
    counts = Counter(tuple(words[i:i + NGRAM_WORDS]) for i in range(len(words) - NGRAM_WORDS + 1))
    if counts:
        phrase, count = counts.most_common(1)[0]
        if count >= MIN_NGRAM_REPEATS:
            flags["repeated_phrase"] = {"count": count, "phrase": " ".join(phrase)}
    lines = Counter(line.strip() for line in text.splitlines() if len(line.strip()) >= 30)
    if lines:
        line, count = lines.most_common(1)[0]
        if count >= 6:
            flags["repeated_line"] = {"count": count, "line": line[:300]}
    if "<think>" in text and "</think>" not in text:
        flags["unclosed_thinking"] = True
    closing_tags = text.count("</think>")
    if closing_tags >= 3:
        flags["repeated_thinking_close_tag"] = closing_tags
    return flags


def audit(result_path, output_dir):
    raw_bytes = result_path.read_bytes()
    result = json.loads(raw_bytes.decode("utf-8", errors="strict"))
    details = result.get("details")
    if not isinstance(details, dict) or not details:
        raise ValueError("No per-sample details: check --dump-eval-details and the results file")
    details = {k: v for k, v in details.items() if str(k).isdigit()}
    if not details:
        raise ValueError("No numeric sample entries in details")
    duplicate_groups = defaultdict(list)
    prompts = defaultdict(list)
    flagged = []
    wrong = []
    lengths = []
    correct = 0
    flag_counts = Counter()
    for index, row in sorted(details.items(), key=lambda item: int(item[0])):
        text = row["origin_prediction"]
        if not isinstance(text, str):
            raise TypeError(f"Unexpected origin_prediction type for {index}: {type(text)}")
        lengths.append(len(text))
        correct += row.get("correct") is True
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) >= MIN_DUPLICATE_CHARS:
            duplicate_groups[hashlib.sha256(normalized.encode(errors="surrogatepass")).hexdigest()].append(index)
        prompts[json.dumps(row.get("prompt"), sort_keys=True, ensure_ascii=False)].append(index)
        flags = inspect_text(text)
        if row.get("predictions") in ("NULL", "", None):
            flags["answer_extraction_failed"] = True
        evidence = {"index": index, "correct": row.get("correct"),
                    "prediction": row.get("predictions"), "reference": row.get("references"),
                    "chars": len(text), "flags": flags, "head": text[:1200], "tail": text[-1600:],
                    "replacement_contexts": [{"position": m.start(), "context": text[max(0, m.start()-60):m.start()+70]}
                                             for m in list(re.finditer("\ufffd", text))[:10]]}
        if flags:
            flagged.append(evidence)
            flag_counts.update(flags.keys())
        if row.get("correct") is not True:
            wrong.append(evidence)
    duplicates = [ids for ids in duplicate_groups.values() if len(ids) > 1]
    duplicate_prompts = [ids for ids in prompts.values() if len(ids) > 1]
    summary = {
        "source": str(result_path), "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "samples": len(details), "correct": correct, "incorrect": len(wrong),
        "reported_accuracy": result.get("accuracy"), "recomputed_accuracy": 100 * correct / len(details),
        "utf8_decode": "strict_success", "flag_counts": dict(flag_counts),
        "flagged_samples": len(flagged),
        "flag_correctness": {name: {"correct": sum(r["correct"] is True for r in flagged if name in r["flags"]),
                                     "incorrect": sum(r["correct"] is not True for r in flagged if name in r["flags"])}
                             for name in flag_counts},
        "duplicate_full_outputs": duplicates,
        "duplicate_prompts": duplicate_prompts,
        "characters": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "limits": "Repetition/mojibake flags are heuristics requiring inspection. This is not a proof of semantic correctness. Numeric final-answer equality is not a duplicate-output flag.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("output-audit.json", summary), ("flagged-samples.json", flagged), ("wrong-samples.json", wrong)):
        (output_dir / name).write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result, args.output_dir)
