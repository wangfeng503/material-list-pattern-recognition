#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病害材料预测器 - 选择最匹配的材料清单模式
根据病害类型和尺寸，从案例库中选择最合适的材料清单

核心改进：
  1. 外挂 config.yaml 配置文件（病害单位、材料单位、变换类型）
  2. 从原始数据加载并按材料组合分组，使用 per-material 变换计算用量比
  3. 支持线性(linear)、二次根(sqrt)、三次根(cbrt)、二次方(square)、三次方(cube)变换

使用方式：
    from 5matching_system import DiseaseMaterialPredictor

    predictor = DiseaseMaterialPredictor()
    result = predictor.predict("车辙", 18.0)
    print(result)
"""

import os
import json
import math
import yaml
import numpy as np
from collections import defaultdict


# ============================================================
# 变换函数
# ============================================================

_TRANSFORM_FUNCS = {
    'linear': lambda s: s,
    'sqrt':   lambda s: math.sqrt(s),
    'cbrt':   lambda s: s ** (1.0 / 3.0),
    'square': lambda s: s ** 2,
    'cube':   lambda s: s ** 3,
}


def transform_size(size, transform_type='linear'):
    """
    根据变换类型对尺寸进行变换

    Args:
        size: 病害尺寸
        transform_type: 变换类型 (linear/sqrt/cbrt/square/cube)

    Returns:
        变换后的尺寸值
    """
    func = _TRANSFORM_FUNCS.get(transform_type, _TRANSFORM_FUNCS['linear'])
    return func(size)


# ============================================================
# 异常值剔除
# ============================================================

def _remove_outliers_zscore(values, threshold=3.0):
    """Z-score异常值剔除"""
    if len(values) < 3:
        return list(values)
    arr = np.array(values, dtype=float)
    z = np.abs((arr - arr.mean()) / (arr.std() + 1e-12))
    filtered = arr[z < threshold]
    return filtered.tolist() if len(filtered) >= 2 else list(values)


# ============================================================
# 预测器
# ============================================================

class DiseaseMaterialPredictor:
    """
    病害材料预测器 - 选择最匹配的材料清单模式

    核心功能：
    1. 从 config.yaml 加载配置（病害单位、材料单位、变换类型）
    2. 从原始数据加载案例，按材料组合分组，使用 per-material 变换计算用量比
    3. 根据病害类型和尺寸选择最匹配的材料清单
    4. 返回结构化的预测结果

    输出格式：
    {
        'success': bool,
        'message': str,
        'disease_type': str,
        'size': float,
        'unit': str,
        'predictions': [
            {
                'materials': list,
                'material_units': list,
                'mean': list,
                'std': list,
                'confidence': float,
                'sample_count': int,
                'frequency_weight': float,
                'area_similarity': float,
                'material_similarity': float,
                'transforms': list  # 每个材料对应的变换类型
            }
        ]
    }
    """

    def __init__(self, config_path='config.yaml', raw_data_dir='dataset/lumian'):
        """
        Args:
            config_path: 配置文件路径
            raw_data_dir: 原始数据目录
        """
        self.config_path = config_path
        self.raw_data_dir = raw_data_dir
        self.config = self._load_config()
        self.disease_units = self.config.get('disease_units', {})
        self.material_units = self.config.get('material_units', {})
        self.transformations = self.config.get('transformations', {})
        self.default_transform = self.config.get('default_transform', 'linear')

        self.disease_data = {}
        self.disease_total_cases = {}
        self._load_data()

    def _load_config(self):
        """加载 config.yaml 配置文件"""
        if not os.path.exists(self.config_path):
            print(f"警告: 配置文件 {self.config_path} 不存在，使用默认配置")
            return {
                'disease_units': {},
                'material_units': {},
                'transformations': {},
                'default_transform': 'linear'
            }
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_transform(self, disease_type, material):
        """获取指定(病害类型, 材料)的变换类型"""
        disease_transforms = self.transformations.get(disease_type, {})
        return disease_transforms.get(material, self.default_transform)

    def _load_data(self):
        """从原始数据加载案例，按材料组合分组并计算统计量"""
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {self.raw_data_dir}")

        for fname in sorted(os.listdir(self.raw_data_dir)):
            if not fname.endswith('.jsonl'):
                continue
            disease_type = fname.replace('.jsonl', '')
            filepath = os.path.join(self.raw_data_dir, fname)

            # 读取所有案例
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

            # 按材料组合分组
            combo_map = {}
            total_valid_cases = 0
            for case in cases:
                codes, amounts, size = self._extract_material_info(case)
                if not codes or size <= 0:
                    continue
                total_valid_cases += 1
                combo_str = ','.join(codes)
                if combo_str not in combo_map:
                    combo_map[combo_str] = {
                        'materials': codes,
                        'sizes': [],
                        'amounts_list': [],
                    }
                combo_map[combo_str]['sizes'].append(size)
                combo_map[combo_str]['amounts_list'].append(amounts)

            # 计算每个组合的统计量
            patterns = []
            for combo_str, info in combo_map.items():
                count = len(info['sizes'])
                materials = info['materials']
                sizes = info['sizes']

                # 面积统计
                cleaned_sizes = _remove_outliers_zscore(sizes)
                mean_area = float(np.mean(cleaned_sizes)) if cleaned_sizes else 0.0
                variance_area = float(np.var(cleaned_sizes, ddof=1)) if len(cleaned_sizes) > 1 else 0.0

                # 每个材料的用量比统计（使用 per-material 变换）
                mean_vector = []
                variance_vector = []
                transforms_used = []

                for j, material in enumerate(materials):
                    transform = self._get_transform(disease_type, material)
                    transforms_used.append(transform)

                    # 收集该材料在所有案例中的用量比 = amount / f(size)
                    ratios = []
                    for i, amounts in enumerate(info['amounts_list']):
                        if j < len(amounts) and amounts[j] > 0 and sizes[i] > 0:
                            transformed_size = transform_size(sizes[i], transform)
                            ratios.append(amounts[j] / transformed_size)

                    if ratios:
                        cleaned_ratios = _remove_outliers_zscore(ratios)
                        mean_val = float(np.mean(cleaned_ratios)) if cleaned_ratios else 0.0
                        var_val = float(np.var(cleaned_ratios, ddof=1)) if len(cleaned_ratios) > 1 else 0.0
                    else:
                        mean_val = 0.0
                        var_val = 0.0

                    mean_vector.append(mean_val)
                    variance_vector.append(var_val)

                percentage = (count / total_valid_cases * 100) if total_valid_cases > 0 else 0

                patterns.append({
                    'materials': materials,
                    'count': count,
                    'percentage': round(percentage, 2),
                    'mean_area': round(mean_area, 4),
                    'variance_area': round(variance_area, 4),
                    'mean': [round(v, 4) for v in mean_vector],
                    'variance': [round(v, 4) for v in variance_vector],
                    'transforms': transforms_used,
                })

            # 按count降序排序
            patterns.sort(key=lambda x: -x['count'])
            self.disease_data[disease_type] = patterns
            self.disease_total_cases[disease_type] = total_valid_cases

        print(f"已加载 {len(self.disease_data)} 种病害类型")

    def _extract_material_info(self, data):
        """
        从单条数据中提取材料编号、用量和面积

        Returns:
            (sorted_codes, amounts, area)
            - codes: 排序后的材料编号列表
            - amounts: 对应的 check_amount（物理用量）
            - area: 病害尺寸 quantity
        """
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

        # 按材料编号排序，保持 codes 和 amounts 的对应关系
        if codes:
            paired = sorted(zip(codes, amounts), key=lambda x: x[0])
            codes = [p[0] for p in paired]
            amounts = [p[1] for p in paired]

        return codes, amounts, area

    def get_disease_types(self):
        """获取所有可用的病害类型"""
        return list(self.disease_data.keys())

    def get_disease_unit(self, disease_type):
        """获取病害类型对应的标准单位"""
        return self.disease_units.get(disease_type, '平方米')

    def _calculate_jaccard_similarity(self, set1, set2):
        """计算两个集合的Jaccard相似度"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _calculate_material_ratio_similarity(self, pattern_materials, pattern_means, target_materials=None, target_amounts=None):
        """计算材料比例相似度"""
        if not pattern_materials or len(pattern_materials) != len(pattern_means):
            return 0.0
        if target_materials is None or target_amounts is None:
            return 1.0

        pattern_ratio_dict = dict(zip(pattern_materials, pattern_means))
        total_pattern = sum(pattern_means) if sum(pattern_means) > 0 else 1

        matched_sum = 0.0
        total_weight = 0.0

        for mat, amount in zip(target_materials, target_amounts):
            if mat in pattern_ratio_dict:
                pattern_ratio = pattern_ratio_dict[mat] / total_pattern
                target_ratio = amount / (sum(target_amounts) if sum(target_amounts) > 0 else 1)
                similarity = 1 - abs(pattern_ratio - target_ratio)
                matched_sum += similarity * amount
                total_weight += amount

        return matched_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_pattern_score(self, pattern, size, disease_type, target_materials=None, target_amounts=None):
        """
        计算模式匹配得分

        评分因素：
        1. 频率加权：模式出现频率
        2. 动态阈值：根据总案例数调整小样本惩罚
        3. 面积相似度：高斯核
        4. 材料组合相似度：Jaccard + 比例
        5. 样本量可信度
        """
        count = pattern['count']
        mean_area = pattern.get('mean_area', 0)
        materials = pattern['materials']
        means = pattern.get('mean', [])

        total_cases = self.disease_total_cases.get(disease_type, 1)

        frequency_weight = count / total_cases

        base_min_count = 5
        dynamic_threshold = base_min_count + math.log(total_cases + 1) * 2
        dynamic_threshold = min(dynamic_threshold, 30)

        if count < dynamic_threshold:
            count_penalty = (count / dynamic_threshold) ** 2
        else:
            count_penalty = 1.0

        area_std = math.sqrt(pattern.get('variance_area', 1)) if pattern.get('variance_area', 0) > 0 else 10.0
        area_distance = abs(size - mean_area)
        area_similarity = math.exp(-(area_distance ** 2) / (2 * area_std ** 2))

        if target_materials:
            mat_set = set(materials)
            target_set = set(target_materials)
            jaccard_sim = self._calculate_jaccard_similarity(mat_set, target_set)
            ratio_sim = self._calculate_material_ratio_similarity(materials, means, target_materials, target_amounts)
            material_similarity = (jaccard_sim + ratio_sim) / 2
        else:
            material_similarity = 1.0

        confidence_boost = 1 + math.log(count + 1) / 10

        score = frequency_weight * area_similarity * count_penalty * material_similarity * confidence_boost

        return score, frequency_weight, area_similarity, count_penalty, material_similarity

    def predict(self, disease_type, size, unit=None, top_n=1, target_materials=None, target_amounts=None):
        """
        根据病害类型和尺寸选择最匹配的材料清单模式

        Args:
            disease_type: 病害类型（如 '车辙', '坑槽'）
            size: 病害尺寸值（如 18.0）
            unit: 尺寸单位，可选，默认自动获取
            top_n: 返回前N个最匹配的模式，默认1
            target_materials: 目标材料列表（用于材料组合相似度计算）
            target_amounts: 目标材料用量列表

        Returns:
            dict: 预测结果
        """
        if disease_type not in self.disease_data:
            return {
                'success': False,
                'message': f"未找到病害类型: {disease_type}",
                'disease_type': disease_type,
                'size': size,
                'unit': unit,
                'predictions': []
            }

        patterns = self.disease_data[disease_type]
        if not patterns:
            return {
                'success': False,
                'message': "该病害类型没有可用的案例模式",
                'disease_type': disease_type,
                'size': size,
                'unit': unit,
                'predictions': []
            }

        standard_unit = self.get_disease_unit(disease_type)
        if unit is None:
            unit = standard_unit

        scored_patterns = []
        for idx, pattern in enumerate(patterns):
            score, freq_weight, area_sim, count_penalty, mat_sim = self._calculate_pattern_score(
                pattern, size, disease_type, target_materials, target_amounts
            )
            scored_patterns.append({
                'idx': idx,
                'pattern': pattern,
                'score': score,
                'frequency_weight': freq_weight,
                'area_similarity': area_sim,
                'count_penalty': count_penalty,
                'material_similarity': mat_sim
            })

        scored_patterns.sort(key=lambda x: -x['score'])

        results = []
        for sp in scored_patterns[:top_n]:
            pattern = sp['pattern']
            transforms = pattern.get('transforms', [])

            means = []
            stds = []
            material_units = []
            transform_list = []

            for i, material in enumerate(pattern['materials']):
                mean_ratio = pattern['mean'][i]
                var_ratio = pattern['variance'][i]
                std_ratio = math.sqrt(var_ratio) if var_ratio > 0 else 0.0

                # 获取该材料的变换类型
                transform = transforms[i] if i < len(transforms) else self._get_transform(disease_type, material)
                transform_list.append(transform)

                # 使用变换后的尺寸计算用量
                transformed_size = transform_size(size, transform)
                mean_usage = mean_ratio * transformed_size
                std_usage = std_ratio * transformed_size

                means.append(round(mean_usage, 4))
                stds.append(round(std_usage, 4))
                material_units.append(self.material_units.get(material, '未知'))

            results.append({
                'materials': pattern['materials'],
                'material_units': material_units,
                'mean': means,
                'std': stds,
                'confidence': round(sp['score'], 4),
                'sample_count': pattern['count'],
                'frequency_weight': round(sp['frequency_weight'], 4),
                'area_similarity': round(sp['area_similarity'], 4),
                'material_similarity': round(sp['material_similarity'], 4),
                'transforms': transform_list
            })

        return {
            'success': True,
            'message': f"预测成功，找到 {len(results)} 个匹配模式",
            'disease_type': disease_type,
            'size': size,
            'unit': unit,
            'predictions': results
        }

    def print_result(self, result):
        """打印预测结果"""
        if not result['success']:
            print(f"\n❌ {result['message']}")
            return

        print(f"\n📊 {result['disease_type']} 病害（尺寸: {result['size']} {result['unit']}）")
        print(f"状态: ✅ 成功")
        print(f"消息: {result['message']}")

        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n--- 匹配模式 {i} (置信度: {pred['confidence']:.4f}) ---")
            print(f"样本数量: {pred['sample_count']}")
            print(f"频率权重: {pred['frequency_weight']:.4f}")
            print(f"面积相似度: {pred['area_similarity']:.4f}")
            print(f"材料相似度: {pred['material_similarity']:.4f}")

            print(f"\n材料清单（共 {len(pred['materials'])} 种材料）:")
            print("-" * 85)
            print(f"{'材料编号':<15} {'单位':<8} {'变换':<10} {'均值':<12} {'标准差':<12} {'均值±标准差'}")
            print("-" * 85)

            for j, material in enumerate(pred['materials']):
                mean = pred['mean'][j]
                std = pred['std'][j]
                unit = pred['material_units'][j]
                transform = pred['transforms'][j] if j < len(pred.get('transforms', [])) else 'linear'
                print(f"{material:<15} {unit:<8} {transform:<10} {mean:<12.4f} {std:<12.4f} [{mean-std:.4f}, {mean+std:.4f}]")


def main():
    """测试用例"""
    print("=" * 60)
    print("病害材料预测器 - 选择最匹配的材料清单")
    print("=" * 60)

    predictor = DiseaseMaterialPredictor()

    test_cases = [
        {"disease_type": "车辙", "size": 18.0},
        {"disease_type": "坑槽", "size": 10.0},
        {"disease_type": "拱起", "size": 5.0},
    ]

    for test_case in test_cases:
        print(f"\n" + "=" * 60)
        print(f"测试: {test_case['disease_type']} 病害, 尺寸: {test_case['size']}")
        print("=" * 60)

        result = predictor.predict(test_case['disease_type'], test_case['size'], top_n=2)
        predictor.print_result(result)


if __name__ == '__main__':
    main()
