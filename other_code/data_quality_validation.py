#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量验证脚本
用于验证处理后的数据质量和完整性
"""

import os
import pandas as pd
import numpy as np
import json

PROCESSED_DATA_FILE = './output/processed_data.csv'
TRAIN_DATA_FILE = './output/train_data.csv'


def validate_data_quality():
    print("开始数据质量验证...")
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"错误: 处理后数据文件 {PROCESSED_DATA_FILE} 不存在")
        return False
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"错误: 训练数据文件 {TRAIN_DATA_FILE} 不存在")
        return False
    try:
        processed_df = pd.read_csv(PROCESSED_DATA_FILE)
        train_df = pd.read_csv(TRAIN_DATA_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False
    print(f"处理后数据行数: {len(processed_df)}")
    print(f"训练数据行数: {len(train_df)}")
    print("\n1. 数据完整性检查:")
    required_cols = ['bill_code', 'bill_price', 'quantity', 'total_amount']
    for col in required_cols:
        if col not in processed_df.columns:
            print(f"警告: 字段 {col} 不存在于处理后数据中")
        else:
            missing_count = processed_df[col].isnull().sum()
            print(f"字段 {col} 缺失值数量: {missing_count}")
    print("\n2. 数据分布检查:")
    if 'bill_price' in processed_df.columns:
        print("价格统计:")
        print(processed_df['bill_price'].describe())
    if 'quantity' in processed_df.columns:
        print("\n数量统计:")
        print(processed_df['quantity'].describe())
    if 'total_amount' in processed_df.columns:
        print("\n金额统计:")
        print(processed_df['total_amount'].describe())
    print("\n3. 数据一致性检查:")
    if all(col in processed_df.columns for col in ['bill_price', 'quantity', 'total_amount']):
        calculated_amount = processed_df['bill_price'] * processed_df['quantity']
        difference = abs(processed_df['total_amount'] - calculated_amount)
        max_diff = difference.max()
        print(f"金额计算最大差异: {max_diff}")
        if max_diff > 0.01:
            print("警告: 金额计算存在较大差异")
        else:
            print("金额计算一致")
    print("\n4. 特征检查:")
    feature_cols = [col for col in train_df.columns if col not in ['total_amount', 'bill_code']]
    print(f"训练数据特征数量: {len(feature_cols)}")
    print(f"训练数据列名: {train_df.columns.tolist()}")
    print("\n5. 分类特征检查:")
    disease_type_cols = [col for col in processed_df.columns if col.startswith('disease_type_')]
    print(f"病害类型数量: {len(disease_type_cols)}")
    if 'bill_code' in processed_df.columns:
        unique_bill_codes = processed_df['bill_code'].nunique()
        print(f"唯一清单编码数量: {unique_bill_codes}")
    print("\n6. 数据量检查:")
    if len(train_df) < 100:
        print("警告: 训练数据量较少，可能影响模型性能")
    else:
        print("训练数据量充足")
    print("\n数据质量验证完成！")
    return True


def generate_data_report():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"错误: 处理后数据文件 {PROCESSED_DATA_FILE} 不存在")
        return
    try:
        df = pd.read_csv(PROCESSED_DATA_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    report = {
        'data_overview': {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'file_size': os.path.getsize(PROCESSED_DATA_FILE) / 1024 / 1024
        },
        'missing_values': {},
        'numeric_stats': {},
        'categorical_stats': {}
    }
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        report['missing_values'][col] = {'count': int(missing_count), 'percent': round(missing_percent, 2)}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report['numeric_stats'][col] = df[col].describe().to_dict()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        top_values = df[col].value_counts().head(5).to_dict()
        report['categorical_stats'][col] = {'unique_count': int(unique_count), 'top_values': top_values}
    report_file = './output/data_quality_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"数据质量报告保存至: {report_file}")
    return report


def main():
    print("开始数据质量验证...")
    validate_data_quality()
    generate_data_report()
    print("数据质量验证和报告生成完成！")


if __name__ == "__main__":
    main()
