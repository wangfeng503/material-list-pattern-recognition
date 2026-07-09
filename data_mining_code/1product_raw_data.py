#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始病害数据加工流水线
======================

本脚本将 CSV 原始数据加工为可用于案例库的 TXT 文本，流程共分三步：

1. CSV 转 JSONL（convert_csv_to_jsonl）
2. 按病害类型拆分 JSONL（split_jsonl_by_disease_type）
3. 生成案例文本（generate_case_text / process_jsonl_file）

使用方式：
    直接运行 `python 1product_raw_data.py` 将依次执行上述三个步骤。
"""

import csv
import io
import json
import os


def convert_csv_to_jsonl(input_file='lumian_data.csv', output_file='lumian_data.jsonl'):
    """将 CSV 文件转换为 JSONL 格式，按照 disease_report_id 分组。"""
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        content = f.read().replace('\r\n', '\n')
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    print(f"读取到 {len(rows)} 条记录")
    grouped_data = {}
    for row in rows:
        disease_report_id = row['disease_report_id']
        if disease_report_id not in grouped_data:
            grouped_data[disease_report_id] = {
                'disease_report_id': disease_report_id,
                'report_time': row['report_time'],
                'complete_time': row['complete_time'],
                'disease_location': row['disease_location'],
                'lon': row['lon'],
                'lat': row['lat'],
                'disease_direction': row['disease_direction'],
                'disease_type_sn': row['disease_type_sn'],
                'quantity': row['quantity'],
                'unit': row['unit'],
                'bill_num': 0,
                'bills': []
            }
        grouped_data[disease_report_id]['bills'].append({
            'bill_code': row['bill_code'],
            'bill_name': row['bill_name'],
            'check_amount': row['check_amount'],
            'bill_unit': row['bill_unit'],
            'bill_price': row['bill_price']
        })
        grouped_data[disease_report_id]['bill_num'] = len(grouped_data[disease_report_id]['bills'])
    result = list(grouped_data.values())
    print(f"分组后共 {len(result)} 个唯一的disease_report_id")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL文件已保存至: {output_file}")


def split_jsonl_by_disease_type(input_file='lumian_data.jsonl', output_dir='lumian'):
    """按照disease_type_sn前两个中文字符分类JSONL文件。"""
    os.makedirs(output_dir, exist_ok=True)
    category_data = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                disease_type = data.get('disease_type_sn', '')
                bills = data.get('bills', [])
                filtered_bills = []
                disease_bill = 0.0
                for bill in bills:
                    check_amount = float(bill.get('check_amount', '0'))
                    bill_price = float(bill.get('bill_price', '0'))
                    if check_amount > 0:
                        filtered_bills.append(bill)
                        disease_bill += check_amount * bill_price
                data['bills'] = filtered_bills
                data['bill_num'] = len(filtered_bills)
                data['disease_bill'] = round(disease_bill, 2)
                if data['bill_num'] == 0:
                    continue
                if len(disease_type) >= 2:
                    category_key = disease_type[:2]
                else:
                    category_key = '其他'
                if category_key not in category_data:
                    category_data[category_key] = []
                category_data[category_key].append(json.dumps(data, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行解析错误: {e}")
                continue
    print(f"共读取 {sum(len(lines) for lines in category_data.values())} 条记录")
    print(f"共分为 {len(category_data)} 个类别")
    for category_key, lines in sorted(category_data.items()):
        safe_name = category_key.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f'{safe_name}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"  {category_key}: {len(lines)} 条 -> {output_file}")
    print(f"\n所有文件已保存至 {output_dir} 目录")


def generate_case_text(item):
    """根据单个JSON数据生成案例文本。"""
    disease_report_id = item.get('disease_report_id', '')
    report_time = item.get('report_time', '')
    disease_location = item.get('disease_location', '')
    disease_type_sn = item.get('disease_type_sn', '')
    disease_direction = item.get('disease_direction', '')
    quantity = item.get('quantity', '')
    unit = item.get('unit', '')
    complete_time = item.get('complete_time', '')
    bill_num = item.get('bill_num', 0)
    bills = item.get('bills', [])
    disease_bill = item.get('disease_bill', 0)
    lines = []
    main_line = (f"根据{disease_report_id}号病害报告，{report_time}发现{disease_location}路段"
                 f"存在{disease_type_sn}病害，方向为{disease_direction}，涉及面积{quantity} {unit}。"
                 f"经处置，{complete_time}完成修复，实施措施包括：")
    lines.append(main_line)
    for i, bill in enumerate(bills, 1):
        bill_code = bill.get('bill_code', '')
        bill_name = bill.get('bill_name', '')
        check_amount = bill.get('check_amount', '')
        bill_unit = bill.get('bill_unit', '')
        bill_price = bill.get('bill_price', '')
        if i < len(bills):
            measure_line = (f"{i}. 按{bill_code}执行{bill_name}"
                            f"（{check_amount} {bill_unit}，单价{bill_price}）；")
        else:
            measure_line = (f"{i}. 按{bill_code}执行{bill_name}"
                            f"（{check_amount} {bill_unit}，单价{bill_price}）。")
        lines.append(measure_line)
    end_line = f"全程共涉及{bill_num}项工程措施，总计{disease_bill}元钱，通过标准化流程恢复路面功能。"
    lines.append(end_line)
    return '\n'.join(lines)


def process_jsonl_file(jsonl_path, output_path):
    """处理单个JSONL文件，生成对应的TXT案例文本。"""
    print(f"\n处理文件: {jsonl_path}")
    jsonl_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    jsonl_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"  跳过解析错误的行: {e}")
    if not jsonl_data:
        print("  没有有效数据，跳过")
        return False
    case_texts = []
    for item in jsonl_data:
        case_text = generate_case_text(item)
        case_texts.append(case_text)
    result = '\n\n'.join(case_texts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"  结果已保存至: {output_path}")
    print(f"  生成内容长度: {len(result)} 字符")
    print(f"  生成案例数: {len(case_texts)}")
    return True


def generate_cases_from_jsonl(input_dir='lumian', output_dir='lumian/txt'):
    """遍历指定目录下的所有 JSONL 文件，生成案例文本。"""
    print("开始处理 lumian 文件夹下的JSONL文件...")
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        return
    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jsonl')])
    if not jsonl_files:
        print("没有找到JSONL文件")
        return
    print(f"共发现 {len(jsonl_files)} 个JSONL文件")
    success_count = 0
    fail_count = 0
    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(input_dir, jsonl_file)
        txt_file = jsonl_file.replace('.jsonl', '.txt')
        output_path = os.path.join(output_dir, txt_file)
        if process_jsonl_file(jsonl_path, output_path):
            success_count += 1
        else:
            fail_count += 1
    print(f"\n处理完成！成功: {success_count} 个, 失败: {fail_count} 个")


def main():
    csv_file = 'lumian_data.csv'
    jsonl_file = 'lumian_data.jsonl'
    split_dir = 'lumian'
    txt_dir = 'lumian/txt'
    print("=" * 60)
    print("步骤一：CSV 转 JSONL")
    print("=" * 60)
    print(f"开始转换 {csv_file} -> {jsonl_file}")
    convert_csv_to_jsonl(csv_file, jsonl_file)
    print("转换完成！\n")
    print("=" * 60)
    print("步骤二：按病害类型拆分 JSONL")
    print("=" * 60)
    print(f"开始按 disease_type_sn 前两个中文字符分类 {jsonl_file}")
    split_jsonl_by_disease_type(jsonl_file, split_dir)
    print("分类完成！\n")
    print("=" * 60)
    print("步骤三：生成案例文本")
    print("=" * 60)
    generate_cases_from_jsonl(split_dir, txt_dir)


if __name__ == "__main__":
    main()
