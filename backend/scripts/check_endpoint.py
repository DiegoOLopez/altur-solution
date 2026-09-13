"""Send dataset calls to a /detect endpoint exactly the way the judge does, and score the answers.

Examples:
    python scripts/check_endpoint.py --url http://127.0.0.1:8000/detect
    python scripts/check_endpoint.py --url https://team.example.com/detect --n 20 --split val

Request body (JSON): {"call_id": "...", "audio_base64": "<base64 of the WAV file bytes>", "sample_rate": 8000, "channels": 2}
Expected response: {"is_synthetic": true|false, "confidence": 0.0-1.0} (confidence optional)

Standard library only. Uses manifest.csv + audio/ from the backend directory by default.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_manifest(path, split):
    with open(path, encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    return rows


def post_call(url, audio_dir, row, timeout):
    path = os.path.join(audio_dir, row["anon_id"] + ".wav")
    with open(path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("ascii")

    body = json.dumps(
        {
            "call_id": row["anon_id"],
            "audio_base64": audio_base64,
            "sample_rate": 8000,
            "channels": 2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}", "latency_s": time.perf_counter() - started}
    except Exception as error:
        return {"error": type(error).__name__, "latency_s": time.perf_counter() - started}

    latency = time.perf_counter() - started
    try:
        data = json.loads(raw)
    except ValueError:
        return {"error": f"HTTP {status}, body is not JSON", "latency_s": latency}

    if not isinstance(data, dict) or not isinstance(data.get("is_synthetic"), bool):
        return {"error": "missing boolean is_synthetic", "latency_s": latency}

    confidence = data.get("confidence")
    if confidence is not None and not (
        isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    ):
        return {"error": "confidence must be a number between 0 and 1", "latency_s": latency}

    return {
        "is_synthetic": data["is_synthetic"],
        "confidence": confidence,
        "latency_s": latency,
    }


def auc(scores, labels):
    """ROC AUC by rank; scores are probabilities of synthetic."""
    pairs = sorted(zip(scores, labels))
    ranks = {}
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        for rank_index in range(index, end):
            ranks[rank_index] = (index + end + 1) / 2
        index = end

    positive_ranks = [
        ranks[index]
        for index, (_, label) in enumerate(pairs)
        if label == 1
    ]
    positive_count = len(positive_ranks)
    negative_count = len(pairs) - positive_count
    if not positive_count or not negative_count:
        return None
    return (
        sum(positive_ranks) - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def score(results):
    answered = [result for result in results if "error" not in result]
    true_positive = sum(
        1
        for result in answered
        if result["label"] == "synthetic" and result["is_synthetic"]
    )
    true_negative = sum(
        1
        for result in answered
        if result["label"] == "human" and not result["is_synthetic"]
    )
    synthetic_count = sum(1 for result in answered if result["label"] == "synthetic")
    human_count = sum(1 for result in answered if result["label"] == "human")
    summary = {
        "calls": len(results),
        "answered": len(answered),
        "errors": len(results) - len(answered),
        "accuracy": (true_positive + true_negative) / len(answered) if answered else None,
        "tpr_synthetic": true_positive / synthetic_count if synthetic_count else None,
        "tnr_human": true_negative / human_count if human_count else None,
        "mean_latency_s": sum(result["latency_s"] for result in results) / len(results) if results else None,
        "max_latency_s": max(result["latency_s"] for result in results) if results else None,
    }
    summary["balanced_accuracy"] = (
        (summary["tpr_synthetic"] + summary["tnr_human"]) / 2
        if synthetic_count and human_count
        else None
    )

    with_confidence = [
        result for result in answered if result["confidence"] is not None
    ]
    if len(with_confidence) == len(answered) and answered:
        probabilities = [
            result["confidence"] if result["is_synthetic"] else 1 - result["confidence"]
            for result in with_confidence
        ]
        labels = [1 if result["label"] == "synthetic" else 0 for result in with_confidence]
        summary["auc"] = auc(probabilities, labels)
        summary["brier"] = sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        ) / len(labels)
    else:
        summary["auc"] = None
        summary["brier"] = None
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="Your /detect endpoint")
    parser.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    parser.add_argument("--audio-dir", default=os.path.join(ROOT, "audio"))
    parser.add_argument("--split", default="val", choices=["train", "val", "all", "hidden"])
    parser.add_argument("--n", type=int, default=10, help="How many calls to send (0 = all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds per call")
    parser.add_argument("--out", help="Write per-call results and summary to this JSON file")
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.split)
    if not rows:
        sys.exit(f"no rows for split {args.split!r} in {args.manifest}")

    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]

    results = []
    for index, row in enumerate(rows, 1):
        result = post_call(args.url, args.audio_dir, row, args.timeout)
        result.update({"call_id": row["anon_id"], "label": row["label"]})
        verdict = result.get("error") or (
            "synthetic" if result["is_synthetic"] else "human"
        )
        mark = "" if "error" in result else (
            "ok " if verdict == row["label"] else "MISS"
        )
        confidence = "" if result.get("confidence") is None else f" conf={result['confidence']:.2f}"
        print(
            f"{index:3d}/{len(rows)} {row['anon_id']} "
            f"truth={row['label']:9s} got={verdict:20s} "
            f"{mark}{confidence} {result['latency_s']:.2f}s",
            flush=True,
        )
        results.append(result)

    summary = score(results)
    print("\nsummary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as output_file:
            json.dump({"summary": summary, "results": results}, output_file, indent=1)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
