#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取真实数据脚本
使用POST请求从接口获取病害计量验收明细表数据
"""

import requests
import json
import os
import csv

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL ="http://10.100.2.98:30404/openApi/api/1828633720382767105/5/os_center/disease_acceptance_detail"

HEADERS = {
    "x-ca-appCodeKey": "svyGm25o",
    "x-ca-appCode": "f08842cb64894a8e07696a8c3bdd0f17abce0a2f",
    "Content-Type": "application/json"
}

fieldnames = ['disease_report_id','report_time','complete_time','disease_location','lon','lat','disease_direction','disease_type_sn','quantity','unit','bill_code','bill_name','check_amount','bill_unit','bill_price']


def fetch_data(page_size=1, page_num=1, verbose=True):
    url = f"{API_URL}?pageSize={page_size}&pageNum={page_num}"
    try:
        if verbose:
            print(f"发送请求到: {url}")
        response = requests.post(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        if verbose:
            print(f"响应状态码: {response.status_code}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def fetch_all_data(page_size=10):
    print("开始获取所有数据...")
    first_page = fetch_data(page_size=page_size, page_num=1, verbose=False)
    if not first_page or 'data' not in first_page:
        print("获取第一页数据失败")
        return None
    total_num = first_page['data'].get('totalNum', 0)
    print(f"接口返回总数据量: {total_num}")
    if total_num == 0:
        print("没有数据")
        return first_page
    all_results = list(first_page['data']['result'])
    print(f"第 1 页获取成功，获取: {len(first_page['data']['result'])} 条，累计: {len(all_results)} 条")
    current_page = 1
    max_pages = 1000
    while len(all_results) < total_num and current_page < max_pages:
        current_page += 1
        print(f"正在获取第 {current_page} 页...")
        page_data = fetch_data(page_size=page_size, page_num=current_page, verbose=False)
        if page_data and 'data' in page_data and 'result' in page_data['data']:
            new_records = page_data['data']['result']
            all_results.extend(new_records)
            print(f"第 {current_page} 页获取成功，新增: {len(new_records)} 条，累计: {len(all_results)} 条")
        else:
            print(f"第 {current_page} 页获取失败，停止请求")
            break
    result = {"code": "200", "msg": "成功", "data": {"pageNum": 1, "pageSize": page_size, "totalNum": len(all_results), "result": all_results}}
    print(f"所有数据获取完成，共获取 {len(all_results)} 条数据")
    return result


def save_to_csv(data, filename="all_data.csv"):
    if not data or 'data' not in data or 'result' not in data['data']:
        print("错误: 没有数据可保存到CSV")
        return False
    records = data['data']['result']
    if not records:
        print("错误: 记录为空，无法保存CSV")
        return False
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV数据已保存至: {filepath}")
    return True


def main():
    print("开始获取所有数据...")
    all_data = fetch_all_data(page_size=100)
    if not all_data:
        print("获取数据失败，程序终止")
        return
    csv_result = save_to_csv(all_data, "all_data.csv")
    if not csv_result:
        print("CSV保存失败，程序终止")
        return
    print("所有数据获取完成！")


if __name__ == "__main__":
    main()
