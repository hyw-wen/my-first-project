import pandas as pd
import numpy as np
import requests
import json
import time
import re
from datetime import datetime
import os

# 配置API信息
API_KEY = "08840522-fe28-47ac-b645-6643ab0af311"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 设置请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def analyze_sentiment_with_llm(text, max_retries=3):
    """
    使用大模型API分析文本情感
    
    参数:
    - text: 要分析的文本
    - max_retries: 最大重试次数
    
    返回:
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感分数（-1到1之间）
    """
    if pd.isna(text) or text.strip() == "":
        return "中性", 0.0
    
    # 限制文本长度，避免超出API限制
    text = text.strip()[:1000]
    
    # 构建提示词
    prompt = f"""请分析以下股吧评论的情感倾向，返回JSON格式的结果：

文本内容：{text}

请按照以下格式返回：
{{
    "sentiment_label": "积极/中性/消极",
    "sentiment_score": 数值（-1到1之间，积极为正，消极为负，中性为0）
}}

请只返回JSON，不要添加其他解释。"""

    # 构建请求数据
    data = {
        "model": "glm-4-flash",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 发送请求
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                
                # 尝试解析JSON
                try:
                    # 提取JSON部分
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        sentiment_data = json.loads(json_str)
                        
                        sentiment_label = sentiment_data.get("sentiment_label", "中性")
                        sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
                        
                        # 验证情感标签
                        if sentiment_label not in ["积极", "中性", "消极"]:
                            sentiment_label = "中性"
                        
                        # 限制分数范围
                        sentiment_score = max(-1.0, min(1.0, sentiment_score))
                        
                        return sentiment_label, sentiment_score
                    else:
                        print(f"无法解析JSON: {content}")
                        return "中性", 0.0
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, 内容: {content}")
                    return "中性", 0.0
            else:
                print(f"API响应格式错误: {result}")
                return "中性", 0.0
                
        except requests.exceptions.RequestException as e:
            print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                return "中性", 0.0
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            return "中性", 0.0
    
    return "中性", 0.0

def batch_analyze_sentiments(comments_df, batch_size=10, save_interval=50):
    """
    批量分析评论情感
    
    参数:
    - comments_df: 包含评论的DataFrame
    - batch_size: 每批处理的评论数量
    - save_interval: 保存中间结果的间隔
    
    返回:
    - 更新后的DataFrame
    """
    # 创建副本以避免修改原始数据
    df = comments_df.copy()
    
    # 初始化新列
    df['llm_sentiment_label_new'] = ""
    df['llm_sentiment_score_new'] = 0.0
    
    total_comments = len(df)
    print(f"开始分析 {total_comments} 条评论的情感...")
    
    # 检查是否有中间结果文件
    temp_file = "temp_sentiment_analysis.csv"
    start_index = 0
    
    if os.path.exists(temp_file):
        try:
            temp_df = pd.read_csv(temp_file)
            if len(temp_df) > 0:
                print(f"发现中间结果文件，从第 {len(temp_df)} 条评论继续...")
                df.update(temp_df)
                start_index = len(temp_df)
        except Exception as e:
            print(f"读取中间结果文件失败: {e}")
    
    # 批量处理
    for i in range(start_index, total_comments, batch_size):
        end_index = min(i + batch_size, total_comments)
        batch = df.iloc[i:end_index]
        
        print(f"处理第 {i+1}-{end_index} 条评论 (共 {total_comments} 条)")
        
        for idx, row in batch.iterrows():
            text = row['combined_text']
            sentiment_label, sentiment_score = analyze_sentiment_with_llm(text)
            
            df.at[idx, 'llm_sentiment_label_new'] = sentiment_label
            df.at[idx, 'llm_sentiment_score_new'] = sentiment_score
            
            print(f"  评论 {idx}: {sentiment_label} ({sentiment_score:.3f})")
        
        # 定期保存中间结果
        if (i + batch_size) % save_interval == 0 or end_index >= total_comments:
            df.iloc[:end_index].to_csv(temp_file, index=False)
            print(f"已保存中间结果到 {temp_file}")
        
        # 添加延迟以避免API限制
        if i + batch_size < total_comments:
            time.sleep(1)
    
    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return df

def update_sentiment_analysis(stock_code="300059"):
    """
    更新情感分析结果
    
    参数:
    - stock_code: 股票代码
    """
    # 读取现有数据
    input_file = f"{stock_code}_sentiment_analysis.csv"
    output_file = f"{stock_code}_sentiment_analysis_updated.csv"
    
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据包含 {len(df)} 条评论")
    
    # 批量分析情感
    updated_df = batch_analyze_sentiments(df)
    
    # 计算融合情感分数（结合新的大模型结果和词典方法）
    updated_df['ensemble_sentiment_score_new'] = (
        0.7 * updated_df['llm_sentiment_score_new'] +  # 大模型权重70%
        0.3 * updated_df['lexicon_sentiment']          # 词典方法权重30%
    )
    
    # 保存更新后的数据
    updated_df.to_csv(output_file, index=False)
    print(f"已保存更新后的情感分析结果到: {output_file}")
    
    # 显示统计信息
    print("\n=== 情感分析统计 ===")
    print(f"总评论数: {len(updated_df)}")
    print(f"积极评论: {(updated_df['llm_sentiment_label_new'] == '积极').sum()} ({(updated_df['llm_sentiment_label_new'] == '积极').sum()/len(updated_df)*100:.1f}%)")
    print(f"中性评论: {(updated_df['llm_sentiment_label_new'] == '中性').sum()} ({(updated_df['llm_sentiment_label_new'] == '中性').sum()/len(updated_df)*100:.1f}%)")
    print(f"消极评论: {(updated_df['llm_sentiment_label_new'] == '消极').sum()} ({(updated_df['llm_sentiment_label_new'] == '消极').sum()/len(updated_df)*100:.1f}%)")
    
    print(f"\n情感分数统计:")
    print(f"新LLM分数 - 均值: {updated_df['llm_sentiment_score_new'].mean():.4f}, 标准差: {updated_df['llm_sentiment_score_new'].std():.4f}")
    print(f"新融合分数 - 均值: {updated_df['ensemble_sentiment_score_new'].mean():.4f}, 标准差: {updated_df['ensemble_sentiment_score_new'].std():.4f}")
    
    return updated_df

if __name__ == "__main__":
    # 运行情感分析
    updated_df = update_sentiment_analysis("300059")
    
    print("\n情感分析完成！")
    print("请检查 300059_sentiment_analysis_updated.csv 文件查看结果。")
