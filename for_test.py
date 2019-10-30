#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Charles
# @Time : 2018/9/23 13:44
# @Email : wcadaydayup@163.com
# @File : for_test.py
# @TODO : 
from datetime import datetime, timedelta
import pandas as pd

def merge_data():
    df = pd.read_csv("data//" + flight_data_csv, encoding="ISO-8859-1")
    l = len(df)
    print(l)
    # print(df)
    # print(df["std"])
    format_str = "%Y/%m/%d %H:%M:%S"
    df["sta_std"] = (pd.to_datetime(df["sta"], format=format_str) - pd.to_datetime(df["std"], format=format_str)).dt.total_seconds()
    # print(df)
    print(df.head())
    print(df.tail(3))
    print(df.sample(4))
    print(df.describe())
    print(df.shape)
    print(df.columns)
    # return
    print(df.info())
    print(df["sta_std"])
    print(df.loc[50])  # 第五十行
    print(df.loc[50:52])  # 这个包含第52个，共三个
    print(df[50:52])  # 这个不包含第52个，共2个
    print(df.loc[[50, 60, 70]])
    # print(df["flight_no"]) #返回一列
    print(df[["flight_no", "departure", "arrival"]])
    print(df.loc[:10, ["departure", "arrival"]])  # a获取前11行dep,arr的数据
    print(df.iloc[:5, 8:])  # 获取前行，9-10列的数据
    import numpy as np
    df["test"] = np.nan
    df["test"] = df["departure"] == "CAN0"  # True or False
    print(df.head())
    df.loc[df["departure"] == df["arrival"], "test"] = "怎么一样了"
    df.loc[df["departure"] != df["arrival"], "test"] = "这还差不多"
    print(df.head())
    df.loc[5:10, "test"] = "我不听"
    print(df.head())
    df.rename(columns={"test": "乱加的一列"}, inplace=True)
    print(df.head())
    data_temp = df[:10]
    data_temp.columns = ["列1", "列2", "列3", "列4", "列5", "列6", "列7", "列8", "列9", "列10"]
    print(df.head())
    data_temp.reindex(columns=["列1", "列4", "列3", "列2", "列9", "列8", "列7", "列68", "列5", "列10"])  # 重新组合列前后顺序
    print(data_temp.head())
    data_temp.reindex(index=[0, 3, 8, 2, 1])  # 默认是按索引排序
    print(data_temp.head())
    data_temp = data_temp.drop([2, 5], axis=0)  # 删除第2、5行
    print(data_temp.head())
    data_temp = data_temp.drop(["列1", "列3"], axis=1)  # 删除第1，3列，axis=1代表删除列
    print(data_temp.head())
    df.loc[df.departure == "-", "departure"] = "这是哪"
    df.sta_std = pd.to_numeric(df.sta_std)
    df.std = pd.to_datetime(df["std"], errors='coerce')
    print(df.std.sample(10))
    print(df[df["sta_std"] > 900].head())
    print(df[(df["sta_std"] > 900 & df["sta_std"] < 5000)])
    print(df[(df["sta_std"] > 900 & df["departure"] == "CAN")])
    print(df.sort_values(by="sta_std", ascending=False).head())
    print(df.sort_values(by=["sta_std", "std"], ascending=False).head())
    print(df["sta_std"].mean())
    print(df["sta_std"].idmax())
    print(df.loc[df["sta_std"].idmax()])
    print(df.sta_std.corr(df.departure))  # 相关性
    df.departur.unique()
    df.departure.value_counts()
    df.dropna()  # 只要某一行中有一个字段是 na 就丢弃
    df.dropna(how='all', inplace=True)  # 丢弃某一行全是na
    df.dropna(axis=1)  # 丢失有na的列
    df.fillna(0)  # 填充数据
    df.fillna({"departure": "XIY", "std": "2018"})
    df["sta_std"].fillna(df["sta_std"].mean())
    df.fillna(method='ffill', limit=3)

    # pd.to_datetime(data[data['last_O_XLMC'] == data['O_XLMC']]['O_SJFCSJ'], format='%H:%M:%S') - pd.to_datetime(
    #     data['last_O_SJFCSJ'], format='%H:%M:%S')).dt.total_seconds()


def time_diff(str_time1, str_time2):
    format_str = "%Y/%m/%d %H:%M:%S"
    t1 = datetime.strptime(str_time1, format_str)
    t2 = datetime.strptime(str_time2, format_str)
    return (t1 - t2).total_seconds()
