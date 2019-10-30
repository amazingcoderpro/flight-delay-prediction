#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Charles
# @Time : 2018/9/20 15:06
# @Email : wcadaydayup@163.com
# @File : main.py
# @TODO : 本代码文件包含了测试数据处理、模型训练、结果预测三个主要功能

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib


ALL_DATA_CSV = "data//all_data.csv"         # 原数据文件, 即《全球程序员节数据_民航.csv》，作了中文字符转英文处理
ALL_DATA_SCORE_CSV = "data//all_data_score.csv"   # 将延误标签、天气量化分数、机型笨重分数、起飞拥堵系数、降落拥堵系系数添加至原始数据上后的数据文件，该文件将作为分类算法的输入数据

SAMPLE_DATA_CSV = "data//sample_data.csv"  # 原数据部分样本文件，从ORIGIN_DATA_CSV中提取进、出港各1000条，共记2000条数据，主要用于样本测试
SAMPLE_DATA_SCORE_CSV = "data//sample_data_score.csv"  # 原数据部分样本文件，从ORIGIN_DATA_CSV中提取进、出港各1000条，共记2000条数据，主要用于样本测试

WEAHTER_CSV = "data//weather.csv"               # 2018年1-6月咸阳机场天气数据文件，来源中国天气网，包含天气、气温、风力等字段
WEATHER_SCORE_CSV = "data//weather_score.csv"   # 带"量化分数"的天气数据文件， 对WEAHTER_CSV中每天的天气情况，通过算法进行量化处理后得到的数据文件，比WEAHTER_CSV多了"SCORE"列,代表当天的天气状况得分，取值范围0-1，分数越高，天气状况越好

AIRCRAFT_CSV = "data//aircraft.csv"              # 原始数据中所有机型参数数据文件，包括飞机长度、宽度、高度、机翼长度、载客量、载货量等参数信息
AIRCRAFT_SCORE_CSV = "data//aircraft_score.csv"     # 带"笨重分数"的机型参数数据文件，对AIRCRAFT_CSV中每种机型的参数通过算法进行量化处理后得到的数据文件，比WEAHTER_CSV多了"SCORE"列，代表该机型的笨重系数, 取值范转0-1，分数越高，代表该机型越笨重

SVM_TRAIN_MODE = "flight_delay_train.pkl"

WEATHER_MAP = {'晴': 1, '阴': 0.9, '多云': 0.8, '小雨': 0.7, '中雨': 0.6, '大雨': 0.4,
               '阵雨': 0.7, '大雪': 0.4, '小雪': 0.7, '中雪': 0.6}
WEATHER_MAP = {'晴': 1, '阴': 0.9, '多云': 0.9, '小雨': 0.7, '中雨': 0.5, '大雨': 0.4,
               '阵雨': 0.7, '大雪': 0.2, '小雪': 0.5, '中雪': 0.4}
WIND_MAP = {"无风": 0, "微风": -0.02, "1级": -0.03, "2级": -0.06, "3级": -0.1, "4级": -0.15, "5级": -0.2, "6级": -0.25}
HIGH_TEMP_LIMIT = 30            # 最高气温
LOW_TEMP_LIMIT = 0              # 最低气温
OUT_TEMP_LIMIT_SCORE = -0.1     # 超限气温扣分


def time2weekday(timestamp):
    """
    根据时间戳计算当前日期属于一周中的哪一天，0--0
    :param timestamp:
    :return: 0--6，代表周天至周六

    """
    return timestamp.dayofweek


def data_process():
    """
    对原始数据进行分析处理，以得到星期数、天气量化分数、机型笨重分数、起飞拥堵系数、降落拥堵系系数五个特征量以及延误标签（0/1)，
    为后面的分类训练提供数据基础
    :return:
    """
    print("start read:{}".format(datetime.now()))
    df = pd.read_csv(SAMPLE_DATA_CSV, encoding="utf-8")
    print(df.shape)
    print(df.info())

    df.dropna(inplace=True)         # 进行处理前，先剔除无效数据
    df.reset_index(inplace=True)

    # 将起飞、降落时间字符串转成datetime类型，方便后面通过时间范围计算起飞、降落拥堵系数
    format_str = "%Y/%m/%d %H:%M:%S"
    df["STD"] = pd.to_datetime(df["STD"], format=format_str)    # 计划起飞时间
    df["STA"] = pd.to_datetime(df["STA"], format=format_str)    # 计划降落时间
    df["ATD"] = pd.to_datetime(df["ATD"], format=format_str)    # 实际起飞时间
    df["ATA"] = pd.to_datetime(df["ATA"], format=format_str)    # 实际降落时间
    df["ATD_STD"] = (df["ATD"] - df["STD"]).dt.total_seconds()    # 起飞延误时间，单位:秒
    df["ATA_STA"] = (df["ATA"] - df["STA"]).dt.total_seconds()    # 降落延误时间，单位:秒
    df['DAYOFWEEK'] = df['STD'].map(time2weekday)

    date2weather_map = get_date2weather_map()
    aircraft2score_map = get_aircraft2score_map()

    flight_no_map = {}
    # 只需要计算从咸阳机场起飞的所有航班的特征数据即可
    total_out = len(df.loc[df["TYPE"] == "OUT"])  # 75574
    for i in range(1, total_out):  # 75560):
        cur_time = df.loc[i, "STD"]     # 当前航班的计划起飞时间
        flight_no = df.loc[i, "FLIGHT_NO"]
        # 获取当前半小时内的计划起飞航班数据
        takeoff_congestion_data = df.loc[(df["TYPE"] == "OUT") & (df["STD"] <= cur_time) &
                                         (df["STD"] >= cur_time - timedelta(seconds=1800))]
        df.loc[i, "TAKEOFF_CONGESTION"] = takeoff_congestion(cur_time, takeoff_congestion_data)

        # 前后各15分钟内计划降落的航班数据
        land_congestion_data = df.loc[(df["TYPE"] == "IN") & (df["ATD"] <= cur_time + timedelta(seconds=900)) &
                                      (df["ATD"] >= cur_time - timedelta(seconds=900))]
        df.loc[i, "LAND_CONGESTION"] = land_congestion(cur_time, land_congestion_data)

        # 得到天气数据得分
        df.loc[i, "WEATHER_SCORE"] = date2weather_map.get(df.loc[i, "STD"].date(), 0)

        # 得到机型笨重系数
        df.loc[i, "AIRCRAFT_SCORE"] = aircraft2score_map.get(df.loc[i, "AIRCRAFT"], aircraft2score_map.get("default", 0))

        df.loc[i, "TIME_PERIOD"] = get_timeperiod(cur_time)

        flight_index = flight_no_map.get(flight_no, None)
        if not flight_index:
            flight_index = len(flight_no_map)
            flight_no_map[flight_no] = len(flight_no_map)

        df.loc[i, "FLIGHT_INDEX"] = flight_index
        df.loc[i, "SEASON"] = get_season(cur_time)

    # 实际起飞时间晚于计划起飞时间大于等于15分钟（即900秒），标记为延误为1，否则为0
    df.loc[df["ATD_STD"] >= 900, "DELAY"] = 1
    df.loc[df["ATD_STD"] < 900, "DELAY"] = 0

    print("{} start write to {} ".format(datetime.now(), ALL_DATA_SCORE_CSV))
    df.to_csv(SAMPLE_DATA_SCORE_CSV, encoding="utf-8")

    print("{} data process finish.".format(datetime.now()))

    takeoff_congestion_mean = df["TAKEOFF_CONGESTION"].mean()
    delay_takeoff_congestion_mean = df.loc[df["DELAY"] == 1, "TAKEOFF_CONGESTION"].mean()

    land_congestion_mean = df["LAND_CONGESTION"].mean()
    delay_land_congestion_mean = df.loc[df["DELAY"] == 1, "LAND_CONGESTION"].mean()

    weather_score_mean = df["WEATHER_SCORE"].mean()
    delay_weather_score_mean = df.loc[df["DELAY"] == 1, "WEATHER_SCORE"].mean()

    aircraft_score_mean = df["AIRCRAFT_SCORE"].mean()
    delay_aircraft_score_mean = df.loc[df["DELAY"] == 1, "AIRCRAFT_SCORE"].mean()

    dayofweek_mean = df["DAYOFWEEK"].mean()
    delay_dayofweek_mean = df.loc[df["DELAY"] == 1, "DAYOFWEEK"].mean()

    print("takeoff_congestion_mean={}, delay_takeoff_congestion_mean={}\n "
          "land_congestion_mean={}, delay_land_congestion_mean={}\n"
          "weather_score_mean={}, delay_weather_score_mean={}\n"
          "aircraft_score_mean={}, delay_aircraft_score_mean={}\n"
          "dayofweek_mean={}, delay_dayofweek_mean={}"
          .format(takeoff_congestion_mean, delay_takeoff_congestion_mean, land_congestion_mean,
                  delay_land_congestion_mean, weather_score_mean, delay_weather_score_mean,
                  aircraft_score_mean, delay_aircraft_score_mean,
                  dayofweek_mean, delay_dayofweek_mean))


def get_season(std_time):
    if (std_time - pd.to_datetime("2018-03-10")).total_seconds() <= 0:
        return 0
    else:
        return 1


def get_timeperiod(std_time):
    """
    获取起飞时间处在一天当中的时间段，从零点开始，每四小时一段，共分4段
    :param std_time:
    :return:
    """
    delta_hours = (std_time - pd.to_datetime(std_time.date())).total_seconds() / 3600
    if delta_hours <= 4:
        return 0
    elif delta_hours <= 8:
        return 1
    elif delta_hours <= 12:
        return 2
    elif delta_hours <= 16:
        return 3
    elif delta_hours <= 20:
        return 4
    else:
        return 5


def takeoff_congestion(current_std, takeoff_data):
    """
    计算某一时间的机场的起飞拥堵系数，即某一时段内计划起飞的航班频度
    :param current_std: 当前航班的计划起飞时间
    :param takeoff_data: 当前航班的前序30分钟内的所有计划起飞的航班数据
    :return: 起飞拥堵系数，越大代表越拥堵
    """
    value = 0
    for std in takeoff_data["STD"]:
        time_delta = (current_std - std).total_seconds()
        if time_delta == 0:
            value += 1
        elif time_delta <= 300:
            value += 0.9
        elif time_delta <= 900:
            value += 0.7
        elif time_delta <= 1800:
            value += 0.5
        elif time_delta <= 3600:
            value += 0.3
        else:
            pass

    return value


def land_congestion(current_std, land_data):
    """
    计算某一时间的机场的降落拥堵系数，即某一时段计划降落航班的频度
    :param current_std: 当前航班的计划起飞时间
    :param land_data: 当前航班的前后序各15分钟内的所有计划降落的航班数据
    :return: 降落拥堵系数，越大代表越拥堵
    """
    value = 0
    for atd in land_data["ATD"]:
        time_delta = (current_std - atd).total_seconds()
        if time_delta == 0:
            value += 1
        elif time_delta <= 300 or time_delta >= -300:
            value += 0.9
        elif time_delta <= 600 or time_delta <= -600:
            value += 0.6
        elif time_delta <= 900 or time_delta <= -900:
            value += 0.3
        else:
            pass

    return value


def get_date2weather_map():
    """
    得到日期与天气数据得分的映射表
    :return:
    """
    df = pd.read_csv(WEATHER_SCORE_CSV, encoding="utf-8")
    format_str = "%Y/%m/%d"
    date2weather_map = {}
    df["DATE"] = pd.to_datetime(df["DATE"], format=format_str)
    for i, row in df.iterrows():
        date2weather_map[row["DATE"].date()] = round(row["SCORE"], 2)

    # print(date2weather_map)
    return date2weather_map


def get_aircraft2score_map():
    """
    得到机型与该机型得分的映射表
    :return: 机型与该机型得分的映射表
    """
    df = pd.read_csv(AIRCRAFT_SCORE_CSV, encoding="utf-8")
    aircraft2score_map = {}
    for i, row in df.iterrows():
        aircraft2score_map[row["AIRCRAFT"]] = row["SCORE"]

    aircraft2score_map["default"] = round(df["SCORE"].mean(), 2)
    # print(aircraft2score_map)
    return aircraft2score_map


def calc_aircraft_score():
    """
    根据机型参数数据计算出该机型的”笨重系数“
    :return:
    """
    df = pd.read_csv(AIRCRAFT_CSV, encoding="utf-8")
    for i, row in df.iterrows():
        volume_all = row["LENGTH"] * row["WIDTH"] * row["HEIGHT"]
        volume_goods = row["GOODS_VOLUME"]
        wing = pd.to_numeric(row["WING"])
        passengers = row["MAX_PASSENGERS"]
        heavy_coefficient = 3 * passengers + 2 * volume_goods + volume_all + 0.5 * wing
        df.loc[i, "HEAVY"] = heavy_coefficient
    max_heavy = df["HEAVY"].max()
    df["SCORE"] = round(df["HEAVY"] / max_heavy, 2)
    df.to_csv(AIRCRAFT_SCORE_CSV, encoding="utf-8")


def calc_weather_score():
    """
    根据天气数据数据量化出该天气状况的”天气分数“
    :return:
    """
    df = pd.read_csv(WEAHTER_CSV, encoding="utf-8")
    for i, row in df.iterrows():
        weather_score = WEATHER_MAP.get(row["WEATHER"], 1)
        weather_score += WIND_MAP.get(row["WIND_L"], 0)
        if float(row["TEMP_H"]) > HIGH_TEMP_LIMIT or float(row["TEMP_L"]) < LOW_TEMP_LIMIT:
            weather_score += OUT_TEMP_LIMIT_SCORE
        df.loc[i, "SCORE"] = round(weather_score, 2)

    df.to_csv(WEATHER_SCORE_CSV, encoding="utf-8")


def future7days_weather_score_map():
    """
    获取最近七天的天气及每天的天气得分
    :return: 返回日期与天气得分的映射表
    """
    from weather import WeatherDataCrawl
    crawl = WeatherDataCrawl()
    weathers = crawl.get_future_7days()
    today = datetime.today()
    weather_score_map = {}
    for i in range(0, 7):
        date = today + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        wea = None
        weather_score = 0.8
        for weather in weathers:
            if date_str == weather.get("DATE"):
                wea = weather.get("WEATHER", "晴")
        if not wea:
            continue

        for k, v in WEATHER_MAP.items():
            if k in wea:
                weather_score = v
        if float(weather.get("TEMP_H", 25)) > HIGH_TEMP_LIMIT or float(weather.get("TEMP_L", 5)) < LOW_TEMP_LIMIT:
            weather_score += OUT_TEMP_LIMIT_SCORE

        wind = weather.get("WIND_L", "无风")
        for k, v in WIND_MAP.items():
            if k in wind:
                weather_score += v
        weather_score_map[date_str] = weather_score

    return weather_score_map


def date2weather_score(date_str):
    """
    根据日期字符串得到该天天气的得分（只支持最近往后的七天内日期）
    :param date_str: 日期字符串，如“2018-10-24”
    :return: 返回该天天气得分
    """
    from weather import WeatherDataCrawl
    crawl = WeatherDataCrawl()
    weathers = crawl.get_future_7days()
    # print(weathers)
    wea = None
    weather_score = 0.8
    for weather in weathers:
        if date_str == weather.get("DATE"):
            wea = weather.get("WEATHER", "晴")
    if not wea:
        return weather_score

    for k, v in WEATHER_MAP.items():
        if k in wea:
            weather_score = v
    if float(weather.get("TEMP_H", 25)) > HIGH_TEMP_LIMIT or float(weather.get("TEMP_L", 5)) < LOW_TEMP_LIMIT:
        weather_score += OUT_TEMP_LIMIT_SCORE

    wind = weather.get("WIND_L", "无风")
    for k, v in WIND_MAP.items():
        if k in wind:
            weather_score += v
    return round(weather_score, 2)


def flight_no2score(flight_no, date_str):
    """
    用以计算指定日期的特定航班的特征量
    :param flight_no: 航班号，如“UH6076”
    :param date_str: 日期字符串，如“2018-10-24”
    :return:
    """
    df = pd.read_csv(ALL_DATA_SCORE_CSV, encoding="utf-8")
    fns = df.loc[df["TYPE"] == "OUT", "FLIGHT_NO"].unique()
    data = df.loc[df["FLIGHT_NO"] == flight_no]
    data.reset_index(inplace=True)
    num = len(data)
    if num == 0:
        return None
    tcs = 0
    lcs = 0
    for i in range(num):
        tcs += data.loc[i, "TAKEOFF_CONGESTION"]
        lcs += data.loc[i, "LAND_CONGESTION"]

    score_map = {"FLIGHT_NO": flight_no,
                 "DATE": date_str,
                 "TAKEOFF_CONGESTION": round(tcs / num, 2),
                 "LAND_CONGESTION": round(lcs / num, 2),
                 "WEATHER_SCORE": date2weather_score(date_str),
                 "AIRCRAFT_SCORE": data.loc[0, "AIRCRAFT_SCORE"]}

    return score_map


def svm_train():
    """利用SVM分类算法，结合量化后的测试数据集进行训练，并保存训练模型"""
    df = pd.read_csv(ALL_DATA_SCORE_CSV)
    df.dropna(inplace=True)  # 只要某一行中有一个字段是 na 就丢弃

    # 取所有数据中的有效的75540条进行训练（即2018-01-01至2018-06-23，6.23日后的降落数据缺失，暂且不用，以提高预测准确率）
    # 其中前5列为待用的特征量列，包括星期，起飞拥堵系数，降落拥堵系数，天气得分，飞机机开笨重系数，起飞时间段，航班号索引，季节，最后一列为标记列，即是否延误，取值 为0、1
    data = df.loc[10:75540, ["TAKEOFF_CONGESTION", "LAND_CONGESTION", "WEATHER_SCORE", "AIRCRAFT_SCORE", "TIME_PERIOD", "FLIGHT_INDEX", "DELAY"]]
    # data = df.loc[10:980,
    #        ["TIME_PERIOD", "AIRCRAFT_SCORE", "FLIGHT_INDEX", "WEATHER_SCORE", "DELAY"]]
    # print(data.isnull().any())
    # delay_null = pd.isnull(data["DELAY"])
    # data_null = data[delay_null==True]

    X, y = np.split(data, (6,), axis=1)     # 取前8列为训练特征列，最后一列为标记列
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)    # 取80%为训练集，20%为验证集

    print("根据测试数据，开始训练...")
    clf = SVC()  # (kernel='linear', decision_function_shape='ovo', C=3)
    clf.fit(x_train, y_train)

    # 训练结果验证
    print("训练完毕，以测试集验证，训练准确率为：{}".format(clf.score(x_test, y_test)))
    # print(clf.predict(x_test))
    # print(y_test)

    # 保存训练模型flight_delay_train.pkl
    joblib.dump(clf, SVM_TRAIN_MODE)


def predict_7days():
    """
    预测最近7天咸阳机场的所有航班的整体起飞延误率
    :return:
    """
    # 读入之前准备好带各项特征量的数据
    weather_score_map = future7days_weather_score_map()
    df = pd.read_csv(ALL_DATA_SCORE_CSV, encoding="utf-8")
    df.dropna(inplace=True)     # 预测之前去掉Nan数据

    fns = df.loc[df["TYPE"] == "OUT", "FLIGHT_NO"].unique()
    today = datetime.today()
    all_flights_data = []

    print("通过原始数据统计分析，结合天气、机型等数据生成未来7天所有航班数据集...")
    # 遍历最近七天，包含当天
    for i in range(0, 7):
        date = today + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        dayofweek = int(date.strftime("%w"))    # 当天属于星期几

        # 遍历7天内所有航班，计算每一趟航班的特征量，为后面的预测做准备
        for flight_no in fns:
            data = df.loc[df["FLIGHT_NO"] == flight_no]
            data.reset_index(inplace=True)
            sum_takeoff_congestion = 0       # 该航班号的历史起飞拥堵系数之和
            sum_land_congestion = 0          # 该航班号的历史降落拥堵系数之和
            sum_takeoff_congestion_week = 0      # 该航班在历史上同一星期几的起飞拥堵系数之和
            sum_land_congestion_week = 0         # 该航班在历史上同一星期几的降落拥堵系数之和
            num_total = len(data)
            num_common_week = 0
            for j in range(num_total):
                if dayofweek == data.loc[j, "DAYOFWEEK"]:
                    sum_takeoff_congestion_week += data.loc[j, "TAKEOFF_CONGESTION"]
                    sum_land_congestion_week += data.loc[j, "LAND_CONGESTION"]
                    num_common_week += 1
                sum_takeoff_congestion += data.loc[j, "TAKEOFF_CONGESTION"]
                sum_land_congestion += data.loc[j, "LAND_CONGESTION"]

            takeoff_congestion = round(sum_takeoff_congestion_week / num_common_week, 2) if sum_takeoff_congestion_week > 0 \
                else round(sum_takeoff_congestion / num_total, 2)
            land_congestion = round(sum_land_congestion_week / num_common_week, 2) if sum_land_congestion_week > 0 \
                else round(sum_land_congestion / num_total, 2)

            season = 1
            # 每一条航班的特征量和测试数据一致，有起飞拥堵系数， 降落拥堵系数， 天气系数， 飞机型号笨重系数,
            # 起飞时间段，航班号索引
            all_flights_data.append(
                [takeoff_congestion, land_congestion,
                 weather_score_map.get(date_str, 0.8), data.iloc[0]["AIRCRAFT_SCORE"],
                 data.iloc[0]["TIME_PERIOD"], data.iloc[0]["FLIGHT_INDEX"]])

    all_flights_data.pop(0)
    # print(all_flights_data)
    print("未来7天航班特征数据生成完毕，共计{}条。".format(len(all_flights_data)))
    print("根据事先训练好的分类模型，开始预测...")
    predict_data = np.array(all_flights_data)
    clf = joblib.load(SVM_TRAIN_MODE)
    result = clf.predict(predict_data)

    score = 0
    for r in result:
        score += r
    ontime_ratio = ((len(result) - score) / len(result))
    print("预测结束，咸阳机场未来七天起飞准点率是：{} %".format(round(ontime_ratio * 100, 2)))
    return ontime_ratio


def check_corr():
    """
    测试特征量与结果的相关性，训练前的测试代码
    :return:
    """
    # 读入之前准备好带各项特征量的数据
    # df = pd.read_csv(SAMPLE_DATA_SCORE_CSV, encoding="utf-8")
    df = pd.read_csv(ALL_DATA_SCORE_CSV, encoding="utf-8")
    df.dropna(inplace=True)     # 预测之前去掉Nan数据
    # print(len(df))
    # format_str = "%Y/%m/%d %H:%M:%S"
    # df["STD"] = pd.to_datetime(df["STD"], format=format_str)    # 计划起飞时间
    # df["STA"] = pd.to_datetime(df["STA"], format=format_str)    # 计划降落时间
    # df["STA_STD"] = (df["STA"] - df["STD"]).dt.total_seconds()

    df_delay = df[df["DELAY"] == 1]
    print("根据给定的测试数据统计出的准点率是：", round(1 - len(df_delay) / len(df), 2))

    # 只看从咸阳机场起飞的
    df_out = df[df["TYPE"] == "OUT"]
    print("TIME_PERIOD:", df_out.TIME_PERIOD.corr(df.DELAY))  # 相关性
    print("AIRCRAFT_SCORE:", df_out.AIRCRAFT_SCORE.corr(df.DELAY))  # 相关性
    print("FLIGHT_INDEX:", df_out.FLIGHT_INDEX.corr(df.DELAY))  # 相关性

    print("TAKEOFF_CONGESTION:", df_out.TAKEOFF_CONGESTION.corr(df.DELAY))  # 相关性
    print("LAND_CONGESTION:", df_out.LAND_CONGESTION.corr(df.DELAY))  # 相关性
    print("WEATHER_SCORE:", df_out.WEATHER_SCORE.corr(df.DELAY))  # 相关性
    # print("SEASON:", df_out.SEASON.corr(df.DELAY))
    # print("DAYOFWEEK:", df_out.DAYOFWEEK.corr(df.DELAY))

    # print(df_out.DAYOFWEEK.value_counts())
    # 统计测试数据中延误数据中的各特征量分布情况
    df2 = df[(df["DELAY"] == 1) & (df["TYPE"] == "OUT")]
    print(df2.DAYOFWEEK.value_counts())
    print(df2.TIME_PERIOD.value_counts())
    print(df2.WEATHER_SCORE.value_counts())

    return


if __name__ == '__main__':
    # calc_aircraft_score()
    # get_aircraft2score_map()

    # calc_weather_score()
    # get_date2weather_map()

    # data_process()
    # 执行此方法开始训练，已经提前训练好模型flight_delay_train.pkl
    # svm_train()
    # 执行此方法开始预测
    predict_7days()

    # test_corr()
