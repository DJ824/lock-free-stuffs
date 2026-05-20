#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LATENCY_RE = re.compile(r"^\s*(.+?):\s+([0-9.]+)\s+sec/round-trip$")
THROUGHPUT_RE = re.compile(r"^\s*(.+?),\s*1,([si]):\s+([0-9,]+)\s+msg/sec$")


def parse_latency_lines(text: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in text.splitlines():
        match = LATENCY_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        seconds = float(match.group(2))
        rows.append((name, seconds * 1e9))
    if not rows:
        raise ValueError("No latency lines found in input")
    return rows


def parse_throughput_lines(text: str) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        match = THROUGHPUT_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        placement = match.group(2)
        msg_per_sec = int(match.group(3).replace(",", ""))
        rows.setdefault(name, {})[placement] = msg_per_sec
    if not rows:
        raise ValueError("No throughput lines found in input")
    return rows


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def plot_latency(rows: list[tuple[str, float]], output_dir: pathlib.Path) -> None:
    names = [name for name, _ in rows]
    latencies_ns = [value for _, value in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f4e79" if "AtomicQueue2" not in name else "#b33a3a" for name in names]
    ax.bar(names, latencies_ns, color=colors)
    ax.set_title("queue_bench2 Ping-Pong Latency")
    ax.set_ylabel("Round-trip latency (ns)")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_latency_bar.png", dpi=180)
    plt.close(fig)

    sorted_rows = sorted(rows, key=lambda item: item[1])
    sorted_names = [name for name, _ in sorted_rows]
    sorted_latencies = [value for _, value in sorted_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(sorted_names, sorted_latencies, color="#2f7d32")
    ax.set_title("queue_bench2 Ping-Pong Latency (Sorted)")
    ax.set_xlabel("Round-trip latency (ns)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_latency_sorted.png", dpi=180)
    plt.close(fig)


def plot_throughput(rows: dict[str, dict[str, int]], output_dir: pathlib.Path) -> None:
    names = list(rows.keys())
    s_values = [rows[name].get("s", math.nan) / 1e6 for name in names]
    i_values = [rows[name].get("i", math.nan) / 1e6 for name in names]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.bar(x - width / 2, s_values, width, label="s", color="#1f4e79")
    ax.bar(x + width / 2, i_values, width, label="i", color="#7a9e9f")
    ax.set_title("queue_bench2 Throughput")
    ax.set_ylabel("Message rate (million msg/sec)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_throughput_grouped.png", dpi=180)
    plt.close(fig)

    best_rows = sorted(
        ((name, max(placements.values())) for name, placements in rows.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    best_names = [name for name, _ in best_rows]
    best_values = [value / 1e6 for _, value in best_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(best_names, best_values, color="#2f7d32")
    ax.invert_yaxis()
    ax.set_title("queue_bench2 Throughput (Best of s/i)")
    ax.set_xlabel("Message rate (million msg/sec)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_throughput_sorted.png", dpi=180)
    plt.close(fig)


def plot_combined(
    throughput_rows: dict[str, dict[str, int]],
    latency_rows: list[tuple[str, float]],
    output_dir: pathlib.Path,
) -> None:
    latency_map = dict(latency_rows)
    combined = []
    for name, placements in throughput_rows.items():
        if name not in latency_map:
            continue
        combined.append((name, max(placements.values()), latency_map[name]))

    if not combined:
        raise ValueError("No overlapping throughput/latency series found")

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [item[1] / 1e6 for item in combined]
    ys = [item[2] for item in combined]
    colors = ["#b33a3a" if name == "AtomicQueue2" else "#1f4e79" for name, _, _ in combined]
    ax.scatter(xs, ys, s=90, c=colors)
    for name, x, y in combined:
        ax.annotate(name, (x / 1e6, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_title("queue_bench2 Throughput vs Latency")
    ax.set_xlabel("Best throughput (million msg/sec)")
    ax.set_ylabel("Round-trip latency (ns)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_combined_scatter.png", dpi=180)
    plt.close(fig)

    names = [item[0] for item in combined]
    throughput_norm = np.array([item[1] for item in combined], dtype=float)
    throughput_norm /= throughput_norm.max()
    latency_norm = np.array([item[2] for item in combined], dtype=float)
    latency_norm = latency_norm.min() / latency_norm

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.bar(x - width / 2, throughput_norm, width, label="throughput score", color="#1f4e79")
    ax.bar(x + width / 2, latency_norm, width, label="latency score", color="#c17c00")
    ax.set_title("queue_bench2 Normalized Throughput/Latency")
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "queue_bench2_combined_normalized.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    text = args.input.read_text()
    latency_rows = parse_latency_lines(text)
    throughput_rows = parse_throughput_lines(text)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(latency_rows, args.output_dir)
    plot_throughput(throughput_rows, args.output_dir)
    plot_combined(throughput_rows, latency_rows, args.output_dir)


if __name__ == "__main__":
    main()
