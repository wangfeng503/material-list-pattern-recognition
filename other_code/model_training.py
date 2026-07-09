#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本
用于训练养护病害金额预测和材料清单号预测模型
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

TRAIN_DATA_FILE = './output/train_data.csv'
MODEL_DIR = './models'
AMOUNT_MODEL_FILE = os.path.join(MODEL_DIR, 'amount_prediction_model.joblib')
BILL_CODE_MODEL_FILE = os.path.join(MODEL_DIR, 'bill_code_prediction_model.joblib')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"错误: 训练数据文件 {TRAIN_DATA_FILE} 不存在")
        return None
    try:
        df = pd.read_csv(TRAIN_DATA_FILE)
        return df
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def prepare_data(df):
    amount_features = [col for col in df.columns if col not in ['total_amount', 'bill_code']]
    amount_target = 'total_amount'
    bill_code_features = [col for col in df.columns if col not in ['bill_code', 'total_amount']]
    bill_code_target = 'bill_code'
    X_amount = df[amount_features]
    y_amount = df[amount_target]
    X_bill_code = df[bill_code_features]
    y_bill_code = df[bill_code_target]
    X_amount_train, X_amount_test, y_amount_train, y_amount_test = train_test_split(X_amount, y_amount, test_size=0.2, random_state=42)
    X_bill_code_train, X_bill_code_test, y_bill_code_train, y_bill_code_test = train_test_split(X_bill_code, y_bill_code, test_size=0.2, random_state=42)
    return {
        'amount': {'X_train': X_amount_train, 'X_test': X_amount_test, 'y_train': y_amount_train, 'y_test': y_amount_test, 'features': amount_features},
        'bill_code': {'X_train': X_bill_code_train, 'X_test': X_bill_code_test, 'y_train': y_bill_code_train, 'y_test': y_bill_code_test, 'features': bill_code_features}
    }


def train_amount_prediction_model(X_train, y_train):
    print("训练金额预测模型...")
    pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(random_state=42))])
    if len(X_train) < 20:
        print("数据量较小，使用默认参数训练模型")
        pipeline.fit(X_train, y_train)
        return pipeline
    else:
        param_grid = {'regressor__n_estimators': [100, 200, 300], 'regressor__max_depth': [None, 10, 20, 30], 'regressor__min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_}")
        return grid_search.best_estimator_


def train_bill_code_prediction_model(X_train, y_train):
    print("训练清单号预测模型...")
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))])
    if len(X_train) < 20:
        print("数据量较小，使用默认参数训练模型")
        pipeline.fit(X_train, y_train)
        return pipeline
    else:
        param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5]}
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_}")
        return grid_search.best_estimator_


def evaluate_amount_model(model, X_test, y_test):
    print("评估金额预测模型...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"均方误差 (MSE): {mse}")
    print(f"均方根误差 (RMSE): {rmse}")
    print(f"R² 评分: {r2}")
    return {'mse': mse, 'rmse': rmse, 'r2': r2}


def evaluate_bill_code_model(model, X_test, y_test):
    print("评估清单号预测模型...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy}")
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    return {'accuracy': accuracy}


def save_model(model, file_path):
    try:
        joblib.dump(model, file_path)
        print(f"模型保存至: {file_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")


def load_model(file_path):
    try:
        model = joblib.load(file_path)
        print(f"模型加载自: {file_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def main():
    print("开始模型训练...")
    print("1. 加载训练数据...")
    df = load_data()
    if df is None:
        print("数据加载失败，退出")
        return
    print("2. 准备训练数据...")
    data = prepare_data(df)
    print("\n3. 训练金额预测模型...")
    amount_model = train_amount_prediction_model(data['amount']['X_train'], data['amount']['y_train'])
    print("\n4. 评估金额预测模型...")
    amount_metrics = evaluate_amount_model(amount_model, data['amount']['X_test'], data['amount']['y_test'])
    print("\n5. 训练清单号预测模型...")
    bill_code_model = train_bill_code_prediction_model(data['bill_code']['X_train'], data['bill_code']['y_train'])
    print("\n6. 评估清单号预测模型...")
    bill_code_metrics = evaluate_bill_code_model(bill_code_model, data['bill_code']['X_test'], data['bill_code']['y_test'])
    print("\n7. 保存模型...")
    save_model(amount_model, AMOUNT_MODEL_FILE)
    save_model(bill_code_model, BILL_CODE_MODEL_FILE)
    print("\n模型训练完成！")
    print(f"金额预测模型性能: R² = {amount_metrics['r2']:.4f}")
    print(f"清单号预测模型性能: 准确率 = {bill_code_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
