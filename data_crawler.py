import sys
import os
# 添加helper_function.py所在目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lecture4_text_data_download'))

from helper_function import download_page, parse_eastmoney_guba
import pandas as pd
import time
import random

# 定义股票代码
stock_code = "300059"  # 东方财富（创业板股票）

# 动态爬取页数
first_page_url = f"http://guba.eastmoney.com/list,{stock_code},f_1.html"
first_page_html = download_page(first_page_url)
total_pages = 100  # 默认值

if first_page_html:
    first_page_data = parse_eastmoney_guba(first_page_html)
    if first_page_data:
        print("第一页数据结构:", list(first_page_data.keys()))
        # 检查是否有文章列表数据
        if 're' in first_page_data:
            article_count = len(first_page_data['re'])
            print(f"每页文章数量: {article_count}")
            # 计算需要爬取的总页数（假设目标是获取1000条评论）
            target_comments = 1000
            total_pages = (target_comments // article_count) + 1
            # 限制最大页数
            total_pages = min(total_pages, 200)
        elif 'article_list' in first_page_data:
            article_count = len(first_page_data['article_list'])
            print(f"每页文章数量: {article_count}")
            target_comments = 1000
            total_pages = (target_comments // article_count) + 1
            total_pages = min(total_pages, 200)

print(f"计划爬取页数: {total_pages}")

# 自适应休眠时间
def get_adaptive_sleep_time(response_time):
    if response_time < 1:
        return random.uniform(1, 2)
    elif response_time < 3:
        return random.uniform(2, 4)
    else:
        return random.uniform(4, 6)

# 构建URL并爬取数据
all_comments = []
for page in range(1, total_pages + 1):
    url = f"http://guba.eastmoney.com/list,{stock_code},f_{page}.html"
    print(f"正在爬取第 {page}/{total_pages} 页: {url}")
    
    start_time = time.time()
    html_content = download_page(url)
    response_time = time.time() - start_time
    
    if html_content:
        comments_data = parse_eastmoney_guba(html_content)
        if comments_data:
            # 检查数据结构
            if 're' in comments_data:
                page_comments = comments_data['re']
                all_comments.extend(page_comments)
                print(f"第 {page} 页成功获取 {len(page_comments)} 条评论")
            elif 'article_list' in comments_data:
                page_comments = comments_data['article_list']
                all_comments.extend(page_comments)
                print(f"第 {page} 页成功获取 {len(page_comments)} 条评论")
            else:
                print(f"第 {page} 页未找到评论数据，数据结构: {list(comments_data.keys())}")
    else:
        print(f"第 {page} 页下载失败")
    
    # 自适应休眠
    sleep_time = get_adaptive_sleep_time(response_time)
    print(f"休眠 {sleep_time:.2f} 秒")
    time.sleep(sleep_time)

# 转换为DataFrame并保存
if all_comments:
    print(f"共获取到 {len(all_comments)} 条评论")
    comments_df = pd.DataFrame(all_comments)
    # 保存为CSV文件，使用UTF-8编码以支持中文
    output_file = f"{stock_code}_comments.csv"
    comments_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {output_file}")
    # 显示数据的前几行和列名
    print("数据列名:", comments_df.columns.tolist())
    print("数据前5行:")
    print(comments_df.head())
else:
    print("未获取到任何评论数据")
