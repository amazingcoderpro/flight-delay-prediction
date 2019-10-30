#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Charles
# @Time : 2018/9/19 16:09
# @Email : wcadaydayup@163.com
# @File : weather.py
# @TODO : 从中国天气网抓取咸阳机场历史以及未来七天的天气数据


import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas.core.frame import DataFrame


class WeatherDataCrawl:
    def __init__(self):
        self.history_index_url = "http://lishi.tianqi.com/xianyang/index.html"
        self.future_7days_url = "http://www.weather.com.cn/weather/101110200.shtml"

    def get_url(self):
        # 获取所有月份的url
        html = requests.get(self.history_index_url).text
        soup = BeautifulSoup(html, 'lxml')  # 解析文档
        all_li = soup.find('div', class_='tqtongji1').find_all('li')
        url_list = []
        for li in all_li:
            url_list.append([li.get_text(), li.find('a')['href']])
        return url_list

    def get_month_weather(self, year_number, month_number):
        """
        获取某年某月的天气数据
        :param year_number:
        :param month_number:
        :return:
        """
        url_list = self.get_url()
        month_url = ""
        for i in range(len(url_list) - 1, -1, -1):
            year_split = int(url_list[i][0].encode('utf-8')[:4])
            month_split = int(url_list[i][0].encode('utf-8')[7:9])
            if year_split == year_number and month_split == month_number:
                month_url = url_list[i][1]

        if not month_url:
            print("未取得{}年{}月的历史数据页面。".format(year_number, month_number))
            return None

        html = requests.get(month_url).text
        soup = BeautifulSoup(html, 'lxml')  # 解析文档
        all_ul = soup.find('div', class_='tqtongji2').find_all('ul')
        month_weather = []
        for i in range(1, len(all_ul)):
            ul = all_ul[i]

            # 將每一天的天气数据以字典存储
            # lis = ul.find_all('li')
            # weather = {
            #     "date": lis[0].get_text(),
            #     "wea": lis[1].get_text(),
            #     "high_tem": lis[2].get_text(),
            #     "low_tem": lis[3].get_text(),
            #     "wind_d": lis[4].get_text(),
            #     "wind_l": lis[5].get_text(),
            # }

            # 将每一天的数据以list存储
            li_list = []
            for li in ul.find_all('li'):
                li_list.append(li.get_text())

            month_weather.append(li_list)
        return month_weather

    def get_year_weather(self, year_number):
        year_weather = []
        for i in range(12):
            month_weather = self.get_month_weather(year_number, i + 1)
            if not month_weather:
                break
            year_weather.extend(month_weather)
            print('%d 年第%d月天气数据采集完成，望您知悉！' % (year_number, i + 1))
        col_name = ['Date', 'Max_Tem', 'Min_Tem', 'Weather', 'Wind', 'Wind_Level']
        result_df = pd.DataFrame(year_weather)
        result_df.columns = col_name
        result_df.to_csv('{}.csv'.format(year_number))
        return result_df

    def get_future_7days(self):
        """
        抓取未来七天咸阳机场的天气数据
        :return:
        """
        today = datetime.datetime.now()
        response = requests.get(self.future_7days_url)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, 'lxml')  # 解析文档
        all_ul = soup.find('div', id='7d').find_all('ul')
        weathers = []
        delta = 0
        for li in all_ul[0].find_all('li'):
            str_date = (today + datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
            ps = li.find_all('p')
            weather = {"DATE": str_date}
            for p in ps:
                if p["class"][0] == "wea":
                    weather["WEATHER"] = p.get_text().strip()
                elif p["class"][0] == "tem":
                    temp_list = p.get_text().split("/")
                    weather["TEMP_H"] = temp_list[0].strip().replace("℃", "")
                    weather["TEMP_L"] = temp_list[-1].strip().replace("℃", "")
                elif p["class"][0] == "win":
                    weather["WIND_D"] = p.find("span")["title"]
                    weather["WIND_L"] = p.get_text().strip()
            weathers.append(weather)
            delta += 1
        return weathers


def get_xianyang_history_weather():
    """
    抓取咸阳机声2018年前6个月的天气数据
    :return:
    """
    weather_result = []
    weather_result.append(["DATE", "TEMP_H", "TEMP_L", "WEATHER", "WIND_D", "WIND_L"])
    crawl = WeatherDataCrawl()
    for i in range(1, 7):
        weather_result += crawl.get_month_weather(2018, i)

    df = DataFrame(weather_result)
    df.to_csv("data//咸阳历史天气数据(201801-201806).csv")


if __name__ == '__main__':
    crawl = WeatherDataCrawl()
    future_7days_weather = crawl.get_future_7days()

    print("未来7天咸阳机场的天气是: ")
    for weather in future_7days_weather:
        print(weather)
