#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 5matching_system.py 中变换的对应关系：
  - 训练侧：pattern.mean[i] == mean(amount_i / f(size))（f 为 config 中该对的变换）
  - 预测侧：predicted[i] == pattern.mean[i] * f(query_size)
"""

import os
import sys
import json
import math
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
matching = import_module('5matching_system')
transform_size = matching.transform_size


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_raw(data_dir='dataset/lumian'):
    data = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.jsonl'):
            continue
        dt = fname.replace('.jsonl', '')
        cases = []
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        data[dt] = cases
    return data


def remove_outliers_zscore(values, threshold=3.0):
    if len(values) < 3:
        return list(values)
    arr = np.array(values, dtype=float)
    z = np.abs((arr - arr.mean()) / (arr.std() + 1e-12))
    filtered = arr[z < threshold]
    return filtered.tolist() if len(filtered) >= 2 else list(values)


TRANSFORM_FUNCS = {
    'linear': lambda s: s,
    'sqrt':   lambda s: math.sqrt(s),
    'cbrt':   lambda s: s ** (1.0 / 3.0),
    'square': lambda s: s ** 2,
    'cube':   lambda s: s ** 3,
}


def main():
    cfg = load_config()
    transforms_cfg = cfg.get('transformations', {})
    default_transform = cfg.get('default_transform', 'linear')
    raw = load_raw()
    predictor = matching.DiseaseMaterialPredictor()
    print("=" * 90)
    print("验证一：训练侧 pattern.mean[i] == mean(amount_i / f(size))")
    print("=" * 90)
    print(f"{'病害':<6} {'材料':<16} {'config变换':<10} {'pattern变换':<12} {'手动mean':<12} {'pattern.mean':<12} {'一致'}")
    print("-" * 90)
    train_ok = 0
    train_fail = 0
    checked = 0
    for dt, patterns in predictor.disease_data.items():
        dt_transforms = transforms_cfg.get(dt, {})
        cases = raw.get(dt, [])
        for pat in patterns:
            materials = pat['materials']
            pat_transforms = pat.get('transforms', [])
            pat_means = pat.get('mean', [])
            combo_cases = []
            for case in cases:
                codes_amts = []
                for bill in case.get('bills', []):
                    code = bill.get('bill_code', '')
                    try:
                        amt = float(bill.get('check_amount', 0))
                    except (ValueError, TypeError):
                        continue
                    if code and amt > 0:
                        codes_amts.append((code, amt))
                if not codes_amts:
                    continue
                codes_amts.sort(key=lambda x: x[0])
                case_codes = [c for c, _ in codes_amts]
                if case_codes == materials:
                    combo_cases.append((case, codes_amts))
            for j, mat in enumerate(materials):
                cfg_t = dt_transforms.get(mat, default_transform)
                pat_t = pat_transforms[j] if j < len(pat_transforms) else default_transform
                if cfg_t == 'linear':
                    continue
                ratios = []
                for case, codes_amts in combo_cases:
                    try:
                        size = float(case.get('quantity', 0))
                    except (ValueError, TypeError):
                        continue
                    if size <= 0:
                        continue
                    for code, amt in codes_amts:
                        if code == mat:
                            ratios.append(amt / TRANSFORM_FUNCS[cfg_t](size))
                            break
                if not ratios:
                    continue
                cleaned = remove_outliers_zscore(ratios)
                manual_mean = float(np.mean(cleaned))
                pat_mean = pat_means[j] if j < len(pat_means) else None
                checked += 1
                match_cfg = (cfg_t == pat_t)
                match_mean = pat_mean is not None and abs(manual_mean - pat_mean) < 1e-3
                ok = match_cfg and match_mean
                if ok:
                    train_ok += 1
                else:
                    train_fail += 1
                print(f"{dt:<6} {mat:<16} {cfg_t:<10} {pat_t:<12} {manual_mean:<12.4f} {str(pat_mean):<12} n={len(ratios):<3} {'✓' if ok else '✗'}")
                if checked >= 25:
                    break
            if checked >= 25:
                break
        if checked >= 25:
            break
    print("-" * 90)
    print(f"训练侧验证: {train_ok}/{checked} 通过 (另有 linear 对跳过)\n")
    print("=" * 90)
    print("验证二：预测侧 predicted[i] == pattern.mean_ratio[i] * f(query_size)")
    print("=" * 90)
    test_cases = [("坑槽", 10.0), ("车辙", 18.0), ("沉陷", 5.0), ("油类", 50.0)]
    predict_ok = 0
    predict_fail = 0
    for dt, size in test_cases:
        if dt not in predictor.disease_data:
            continue
        result = predictor.predict(dt, size, top_n=1)
        if not result.get('success'):
            continue
        pred = result['predictions'][0]
        pred_materials = pred['materials']
        pred_means = pred['mean']
        pred_transforms = pred.get('transforms', [])
        matched_pattern = None
        for pat in predictor.disease_data[dt]:
            if pat['materials'] == pred_materials:
                matched_pattern = pat
                break
        if matched_pattern is None:
            continue
        ratio_vec = matched_pattern.get('mean', [])
        dt_transforms = transforms_cfg.get(dt, {})
        print(f"\n  {dt} (size={size}) — 置信度={pred['confidence']:.4f}  样本数={pred['sample_count']}")
        print(f"  {'材料':<16} {'config变换':<10} {'存储ratio':<12} {'f(size)':<12} {'手动=ratio*f':<14} {'predict输出':<14} {'一致'}")
        for j, mat in enumerate(pred_materials):
            cfg_t = dt_transforms.get(mat, default_transform)
            pred_t = pred_transforms[j] if j < len(pred_transforms) else default_transform
            ratio_j = ratio_vec[j] if j < len(ratio_vec) else 0.0
            f_size = TRANSFORM_FUNCS[cfg_t](size)
            expected = ratio_j * f_size
            actual = pred_means[j]
            ok = (cfg_t == pred_t) and abs(expected - actual) < 1e-2
            if ok:
                predict_ok += 1
            else:
                predict_fail += 1
            print(f"  {mat:<16} {cfg_t:<10} {ratio_j:<12.4f} {f_size:<12.4f} {expected:<14.4f} {actual:<14.4f} {'✓' if ok else '✗'}")
    print(f"\n预测侧验证: {predict_ok}/{predict_ok + predict_fail} 通过")
    print("\n" + "=" * 90)
    print("结论")
    print("=" * 90)
    all_ok = (train_fail == 0) and (predict_fail == 0)
    if all_ok:
        print("✓ 全部通过：config.yaml 中的变换在训练(ratio计算)与预测(用量计算)两侧完全一致。")
    else:
        print(f"✗ 存在不一致：训练侧失败 {train_fail}，预测侧失败 {predict_fail}，需修复。")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
