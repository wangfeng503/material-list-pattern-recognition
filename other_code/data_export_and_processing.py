#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导出和处理脚本
用于从原始JSON数据中提取和清洗数据，进行特征工程
"""

import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_data(df):
    """数据清洗"""
    print("开始数据清洗...")
    df_cleaned = df.copy()
    numeric_cols = ['quantity', 'bill_price', 'check_amount', 'lon', 'lat']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    if 'quantity' in df_cleaned.columns and 'bill_price' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['quantity', 'bill_price'])
        df_cleaned = df_cleaned[(df_cleaned['quantity'] > 0) & (df_cleaned['bill_price'] > 0)]
    if 'total_amount' not in df_cleaned.columns:
        if 'quantity' in df_cleaned.columns and 'bill_price' in df_cleaned.columns:
            df_cleaned['total_amount'] = df_cleaned['quantity'] * df_cleaned['bill_price']
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"数据清洗完成，剩余 {len(df_cleaned)} 条记录")
    return df_cleaned


def feature_engineering(df):
    """特征工程"""
    print("开始特征工程...")
    df_processed = df.copy()
    if 'disease_type_sn' in df_processed.columns:
        disease_dummies = pd.get_dummies(df_processed['disease_type_sn'], prefix='disease_type')
        df_processed = pd.concat([df_processed, disease_dummies], axis=1)
    if 'disease_direction' in df_processed.columns:
        le = LabelEncoder()
        df_processed['disease_direction_encoded'] = le.fit_transform(df_processed['disease_direction'].fillna('未知'))
    if 'unit' in df_processed.columns:
        le = LabelEncoder()
        df_processed['unit_encoded'] = le.fit_transform(df_processed['unit'].fillna('未知'))
    feature_cols = [col for col in df_processed.columns if col.startswith('disease_type_') or col in ['quantity', 'lon', 'lat', 'disease_direction_encoded', 'unit_encoded']]
    print(f"特征工程完成，生成 {len(feature_cols)} 个特征")
    return df_processed, feature_cols


def generate_train_data(df, feature_cols):
    """生成训练数据"""
    print("开始生成训练数据...")
    train_cols = feature_cols + ['total_amount', 'bill_code']
    train_cols = [col for col in train_cols if col in df.columns]
    train_df = df[train_cols].copy()
    train_df = train_df.dropna()
    print(f"训练数据生成完成，共 {len(train_df)} 条记录")
    return train_df


def main():
    print("开始数据处理流程...")
    input_file = os.path.join(OUTPUT_DIR, 'raw_data.json')
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'data' in data and 'result' in data['data']:
        records = data['data']['result']
        df = pd.DataFrame(records)
        print(f"读取到 {len(df)} 条记录")
    else:
        print("错误: 数据格式不正确")
        return
    df_cleaned = clean_data(df)
    df_processed, feature_cols = feature_engineering(df_cleaned)
    train_df = generate_train_data(df_processed, feature_cols)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False, encoding='utf-8-sig')
    print("数据处理完成！")


if __name__ == "__main__":
    main()
