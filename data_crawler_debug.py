import sys
import os

# 添加helper_function.py所在目录到Python路径
sys.path.append(os.path.abspath('lecture4_text_data_download'))

from helper_function import download_page, parse_eastmoney_guba
import pandas as pd
import time
import random

# 定义股票代码
stock_code = "600036"  # 示例：招商银行

print(f"开始调试股票 {stock_code} 的股吧评论数据爬取...")

# 测试URL
url = f"http://guba.eastmoney.com/list,{stock_code},f_1.html"
print(f"测试URL: {url}")

# 下载页面
html_content = download_page(url)

if html_content:
    print(f"\n页面下载成功，大小: {len(html_content)} 字符")
    
    # 保存样本页面以便分析
    with open("sample_page.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("样本页面已保存到 sample_page.html")
    
    # 查看页面的前500个字符，了解页面结构
    print("\n页面前500个字符:")
    print(html_content[:500])
    print("...")
    
    # 查看页面的后500个字符
    print("\n页面后500个字符:")
    print(html_content[-500:])
    print("...")
    
    # 尝试使用不同的正则表达式模式
    print("\n尝试使用不同的正则表达式模式:")
    
    import re
    import json
    
    # 原始模式
    pattern1 = r'var article_list=({.*?});'
    match1 = re.search(pattern1, html_content, re.DOTALL)
    print(f"原始模式匹配结果: {match1 is not None}")
    
    # 更宽松的模式
    pattern2 = r'article_list\s*=\s*({.*?});'
    match2 = re.search(pattern2, html_content, re.DOTALL)
    print(f"宽松模式匹配结果: {match2 is not None}")
    
    # 查找所有可能包含article_list的行
    print("\n查找包含'article_list'的行:")
    lines = html_content.split('\n')
    for i, line in enumerate(lines):
        if 'article_list' in line:
            print(f"第{i+1}行: {line.strip()}")
    
    # 尝试使用BeautifulSoup解析
    print("\n尝试使用BeautifulSoup解析页面结构:")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 查找script标签
    script_tags = soup.find_all('script')
    print(f"找到 {len(script_tags)} 个script标签")
    
    # 检查script标签中是否包含article_list
    for i, script in enumerate(script_tags):
        script_text = script.get_text()
        if 'article_list' in script_text:
            print(f"\nScript标签 {i+1} 包含article_list")
            print(f"内容长度: {len(script_text)}")
            print(f"前300个字符: {script_text[:300]}...")
            
            # 尝试从这个script标签中提取数据
            match = re.search(r'({.*?});', script_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    print("成功解析JSON数据")
                    print(f"数据类型: {type(data)}")
                    if isinstance(data, dict):
                        print(f"字典键: {list(data.keys())}")
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
else:
    print("页面下载失败")

print("\n调试完成")
