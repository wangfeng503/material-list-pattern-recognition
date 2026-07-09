#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性性验证脚本
验证每种材料用量与病害尺寸之间的关系，为5种变换(linear/sqrt/cbrt/square/cube)计算CV和R²，
以CV最小者为最优变换，并生成 config.yaml 和验证报告。

运行方式: python other_code/4verify_linear.py
"""

import os
import sys
import json
import math
import yaml
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'lumian')
DOC_DIR = os.path.join(PROJECT_ROOT, 'doc')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
REPORT_PATH = os.path.join(DOC_DIR, 'linearity_report.md')

TRANSFORMS = {
    'linear': lambda s: s,
    'sqrt':   lambda s: math.sqrt(s),
    'cbrt':   lambda s: s ** (1.0 / 3.0),
    'square': lambda s: s ** 2,
    'cube':   lambda s: s ** 3,
}

TRANSFORM_LABELS = {
    'linear': '一次方(线性)',
    'sqrt':   '二次根(√x)',
    'cbrt':   '三次根(∛x)',
    'square': '二次方(x²)',
    'cube':   '三次方(x³)',
}

MIN_SAMPLES = 3
CV_QUALITY_THRESHOLD = 0.5


def load_raw_data():
    all_data = {}
    for fname in sorted(os.listdir(RAW_DATA_DIR)):
        if not fname.endswith('.jsonl'):
            continue
        disease_type = fname.replace('.jsonl', '')
        filepath = os.path.join(RAW_DATA_DIR, fname)
        cases = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        all_data[disease_type] = cases
    return all_data


def extract_material_info(data):
    codes = []
    amounts = []
    for bill in data.get('bills', []):
        code = bill.get('bill_code', '')
        try:
            amount = float(bill.get('check_amount', 0))
        except (ValueError, TypeError):
            continue
        if code and amount > 0:
            codes.append(code)
            amounts.append(amount)
    try:
        area = float(data.get('quantity', 0))
    except (ValueError, TypeError):
        area = 0.0
    if codes:
        paired = sorted(zip(codes, amounts), key=lambda x: x[0])
        codes = [p[0] for p in paired]
        amounts = [p[1] for p in paired]
    return codes, amounts, area


def remove_outliers_zscore(values, threshold=3.0):
    if len(values) < 3:
        return list(values)
    arr = np.array(values, dtype=float)
    z = np.abs((arr - arr.mean()) / (arr.std() + 1e-12))
    filtered = arr[z < threshold]
    return filtered.tolist() if len(filtered) >= 2 else list(values)


def extract_pairs(all_data):
    pairs = defaultdict(list)
    for disease_type, cases in all_data.items():
        for case in cases:
            codes, amounts, size = extract_material_info(case)
            if not codes or size <= 0:
                continue
            for code, amount in zip(codes, amounts):
                if amount > 0:
                    pairs[(disease_type, code)].append((size, amount))
    return pairs


def evaluate_transform(sizes, amounts, transform_func):
    transformed_sizes = np.array([transform_func(s) for s in sizes])
    amounts_arr = np.array(amounts, dtype=float)
    ratios = amounts_arr / transformed_sizes
    cleaned_ratios = remove_outliers_zscore(ratios)
    ratios_arr = np.array(cleaned_ratios, dtype=float)
    mean_ratio = float(ratios_arr.mean())
    std_ratio = float(ratios_arr.std(ddof=1)) if len(ratios_arr) > 1 else 0.0
    cv = std_ratio / abs(mean_ratio) if abs(mean_ratio) > 1e-12 else float('inf')
    denominator = float(np.sum(transformed_sizes ** 2))
    if denominator < 1e-12:
        r_squared = 0.0
    else:
        k = float(np.sum(amounts_arr * transformed_sizes) / denominator)
        ss_res = float(np.sum((amounts_arr - k * transformed_sizes) ** 2))
        ss_tot = float(np.sum(amounts_arr ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return cv, r_squared


def main():
    print("=" * 80)
    print("材料用量线性性验证")
    print("=" * 80)
    all_data = load_raw_data()
    print(f"加载 {len(all_data)} 种病害类型")
    pairs = extract_pairs(all_data)
    print(f"共 {len(pairs)} 个(病害,材料)对")
    results = {}
    for (disease, material), points in sorted(pairs.items()):
        if len(points) < MIN_SAMPLES:
            continue
        sizes = [p[0] for p in points]
        amounts = [p[1] for p in points]
        best_transform = 'linear'
        best_cv = float('inf')
        for tname, tfunc in TRANSFORMS.items():
            cv, r2 = evaluate_transform(sizes, amounts, tfunc)
            if cv < best_cv:
                best_cv = cv
                best_transform = tname
        if best_transform != 'linear' and best_cv >= CV_QUALITY_THRESHOLD:
            best_transform = 'linear'
        if disease not in results:
            results[disease] = {}
        results[disease][material] = best_transform
    config = {
        'disease_units': {},
        'material_units': {},
        'default_transform': 'linear',
        'transformations': results,
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"配置文件已生成: {CONFIG_PATH}")
    print("完成！")


if __name__ == '__main__':
    main()
