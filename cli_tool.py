#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行工具
用于获取数据、处理数据和训练模型
"""

import argparse
import json
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_real_data import fetch_data as fetch_real_data
from data_export_and_processing import clean_data, feature_engineering, generate_train_data
from model_training import train_amount_prediction_model, train_bill_code_prediction_model
import pandas as pd

# 输出目录
OUTPUT_DIR = './output'
MODEL_DIR = './models'

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='病害计量验收数据处理和模型训练工具')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # fetch 命令
    fetch_parser = subparsers.add_parser('fetch', help='获取真实数据')
    fetch_parser.add_argument('--page-size', type=int, default=10, help='每页条数')
    fetch_parser.add_argument('--page-num', type=int, default=1, help='页码')
    fetch_parser.add_argument('--output', type=str, default='real_response.json', help='输出文件名')
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='处理数据')
    process_parser.add_argument('--input', type=str, default='real_response.json', help='输入数据文件')
    process_parser.add_argument('--output', type=str, default='train_data.csv', help='输出训练数据文件')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--input', type=str, default='output/train_data.csv', help='训练数据文件')
    train_parser.add_argument('--model-type', type=str, choices=['amount', 'bill_code', 'both'], default='both', help='模型类型')
    
    # pipeline 命令
    pipeline_parser = subparsers.add_parser('pipeline', help='完整流程：获取数据 -> 处理 -> 训练')
    pipeline_parser.add_argument('--page-size', type=int, default=10, help='每页条数')
    pipeline_parser.add_argument('--page-num', type=int, default=1, help='页码')
    
    return parser.parse_args()

def fetch_command(args):
    """
    执行获取数据命令
    """
    print(f"获取数据：page_size={args.page_size}, page_num={args.page_num}")
    data = fetch_real_data(page_size=args.page_size, page_num=args.page_num)
    
    if data:
        output_path = os.path.join(OUTPUT_DIR, args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存至: {output_path}")
    else:
        print("获取数据失败")

def process_command(args):
    """
    执行数据处理命令
    """
    input_path = os.path.join(OUTPUT_DIR, args.input)
    if not os.path.exists(input_path):
        input_path = args.input  # 尝试直接使用路径
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 {input_path} 不存在")
        return
    
    print(f"处理数据：{input_path}")
    
    # 读取数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取result数据
        if 'data' in data and 'result' in data['data']:
            records = data['data']['result']
            df = pd.DataFrame(records)
            print(f"读取到 {len(df)} 条记录")
        else:
            print("错误: 数据格式不正确")
            return
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 数据清洗
    df_cleaned = clean_data(df)
    print(f"清洗后：{len(df_cleaned)} 条记录")
    
    # 特征工程
    df_processed, feature_cols = feature_engineering(df_cleaned)
    print(f"特征工程后：{len(feature_cols)} 个特征")
    
    # 生成训练数据
    train_df = generate_train_data(df_processed, feature_cols)
    print(f"训练数据：{len(train_df)} 条记录")
    
    # 保存数据
    output_path = os.path.join(OUTPUT_DIR, args.output)
    train_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"训练数据已保存至: {output_path}")

def train_command(args):
    """
    执行模型训练命令
    """
    input_path = os.path.join(OUTPUT_DIR, args.input)
    if not os.path.exists(input_path):
        input_path = args.input  # 尝试直接使用路径
    
    if not os.path.exists(input_path):
        print(f"错误: 训练数据文件 {input_path} 不存在")
        return
    
    print(f"训练模型：{input_path}")
    
    # 读取训练数据
    try:
        df = pd.read_csv(input_path)
        print(f"读取到 {len(df)} 条训练数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 准备数据
    from model_training import prepare_data
    data = prepare_data(df)
    
    # 训练模型
    if args.model_type in ['amount', 'both']:
        print("\n训练金额预测模型...")
        amount_model = train_amount_prediction_model(
            data['amount']['X_train'],
            data['amount']['y_train']
        )
        from model_training import evaluate_amount_model, save_model
        evaluate_amount_model(amount_model, data['amount']['X_test'], data['amount']['y_test'])
        save_model(amount_model, os.path.join(MODEL_DIR, 'amount_prediction_model.joblib'))
    
    if args.model_type in ['bill_code', 'both']:
        print("\n训练清单号预测模型...")
        bill_code_model = train_bill_code_prediction_model(
            data['bill_code']['X_train'],
            data['bill_code']['y_train']
        )
        from model_training import evaluate_bill_code_model, save_model
        evaluate_bill_code_model(bill_code_model, data['bill_code']['X_test'], data['bill_code']['y_test'])
        save_model(bill_code_model, os.path.join(MODEL_DIR, 'bill_code_prediction_model.joblib'))

def pipeline_command(args):
    """
    执行完整流程命令
    """
    print("开始完整流程...")
    
    # 1. 获取数据
    print("\n1. 获取真实数据...")
    data = fetch_real_data(page_size=args.page_size, page_num=args.page_num)
    
    if not data:
        print("获取数据失败，流程终止")
        return
    
    # 保存原始数据
    raw_data_path = os.path.join(OUTPUT_DIR, 'raw_data.json')
    with open(raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"原始数据已保存至: {raw_data_path}")
    
    # 2. 处理数据
    print("\n2. 处理数据...")
    records = data['data']['result']
    df = pd.DataFrame(records)
    
    df_cleaned = clean_data(df)
    df_processed, feature_cols = feature_engineering(df_cleaned)
    train_df = generate_train_data(df_processed, feature_cols)
    
    train_data_path = os.path.join(OUTPUT_DIR, 'train_data.csv')
    train_df.to_csv(train_data_path, index=False, encoding='utf-8-sig')
    print(f"训练数据已保存至: {train_data_path}")
    
    # 3. 训练模型
    print("\n3. 训练模型...")
    from model_training import prepare_data
    data = prepare_data(train_df)
    
    # 训练金额预测模型
    print("\n3.1 训练金额预测模型...")
    amount_model = train_amount_prediction_model(
        data['amount']['X_train'],
        data['amount']['y_train']
    )
    from model_training import evaluate_amount_model, save_model
    evaluate_amount_model(amount_model, data['amount']['X_test'], data['amount']['y_test'])
    save_model(amount_model, os.path.join(MODEL_DIR, 'amount_prediction_model.joblib'))
    
    # 训练清单号预测模型
    print("\n3.2 训练清单号预测模型...")
    bill_code_model = train_bill_code_prediction_model(
        data['bill_code']['X_train'],
        data['bill_code']['y_train']
    )
    from model_training import evaluate_bill_code_model, save_model
    evaluate_bill_code_model(bill_code_model, data['bill_code']['X_test'], data['bill_code']['y_test'])
    save_model(bill_code_model, os.path.join(MODEL_DIR, 'bill_code_prediction_model.joblib'))
    
    print("\n完整流程执行完成！")

def main():
    """
    主函数
    """
    args = parse_arguments()
    
    if args.command == 'fetch':
        fetch_command(args)
    elif args.command == 'process':
        process_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'pipeline':
        pipeline_command(args)
    else:
        print("请指定命令，使用 --help 查看帮助")

if __name__ == "__main__":
    main()
