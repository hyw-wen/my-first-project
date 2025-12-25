#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票价格数据获取与收益率计算
"""

import pandas as pd
import tushare as ts
import os

# 设置tushare token
TS_TOKEN = "d60077d056be2a4e38f3c75c18ba317cb41ae04bf734b259ee08fcb6"

def get_stock_price(stock_code="300059.SZ", start_date="20250101", end_date="20251231"):
    """
    获取股票的日K线数据
    :param stock_code: 股票代码（带后缀.SZ或.SH）
    :param start_date: 开始日期，格式YYYYMMDD
    :param end_date: 结束日期，格式YYYYMMDD
    :return: 包含价格数据的DataFrame
    """
    print(f"正在获取股票 {stock_code} 的价格数据...")
    
    # 设置token并初始化pro接口
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    try:
        # 获取日K线数据
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        print(f"成功获取 {len(df)} 条日K线数据")
        
        # 处理数据
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values(by='trade_date')
        
        return df
    except Exception as e:
        print(f"获取股票价格数据失败: {e}")
        return None

def calculate_next_day_return(df):
    """
    计算次日收益率
    :param df: 包含日K线数据的DataFrame
    :return: 包含次日收益率的DataFrame
    """
    print("正在计算次日收益率...")
    
    # 计算次日收益率：(次日收盘价 - 当日收盘价) / 当日收盘价
    df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
    
    return df

def main():
    # 设置股票代码和日期范围
    stock_code = "300059.SZ"  # 东方财富（创业板）
    start_date = "20250101"
    end_date = "20251231"
    
    # 获取股票价格数据
    price_df = get_stock_price(stock_code, start_date, end_date)
    if price_df is None:
        return
    
    # 计算次日收益率
    price_df = calculate_next_day_return(price_df)
    
    # 保存数据
    output_file = f"{stock_code.split('.')[0]}_price_data.csv"
    price_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"价格数据和收益率已保存到: {output_file}")
    
    # 显示数据摘要
    print("\n数据摘要:")
    print(price_df.head())
    print(f"\n数据日期范围: {price_df['trade_date'].min()} 到 {price_df['trade_date'].max()}")
    print(f"次日收益率统计:")
    print(price_df['next_day_return'].describe())

if __name__ == "__main__":
    main()
