import pandas as pd
import numpy as np
import requests
import json
import time
import re
from datetime import datetime
import os

# 百度千帆API配置
API_KEY = "bce-v3/ALTAK-7lYegPWnHYwUYqjP7r0dY/d3084f1055af0f4e716b9b840869da1fb2e90eaf"
SECRET_KEY = "ca94ff8e01194cf896c3592dddd22340"

# 获取access token的URL
TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"

# 百度千帆API端点
QIANFAN_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-8k"

def get_access_token(api_key, secret_key):
    """获取百度千帆的access token"""
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    
    try:
        response = requests.get(TOKEN_URL, params=params)
        if response.status_code == 200:
            result = response.json()
            if "access_token" in result:
                return result["access_token"]
            else:
                print(f"获取access token失败: {result}")
                return None
        else:
            print(f"请求失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"获取access token异常: {str(e)}")
        return None

def analyze_sentiment_with_qianfan(text, access_token, max_retries=3):
    """
    使用百度千帆大模型API分析文本情感
    
    参数:
    - text: 要分析的文本
    - access_token: 百度千帆access token
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
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "system": "你是一个专业的情感分析助手，请准确分析文本的情感倾向。"
    }
    
    params = {"access_token": access_token}
    
    for attempt in range(max_retries):
        try:
            # 添加延迟，避免API频率限制
            time.sleep(0.5)
            
            response = requests.post(
                QIANFAN_URL,
                params=params,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    content = result["result"].strip()
                    
                    # 尝试解析JSON响应
                    try:
                        # 提取JSON部分
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            sentiment_data = json.loads(json_str)
                            
                            sentiment_label = sentiment_data.get("sentiment_label", "中性")
                            sentiment_score = float(sentiment_data.get("sentiment_score", 0))
                            
                            # 确保分数在-1到1之间
                            sentiment_score = max(-1.0, min(1.0, sentiment_score))
                            
                            return sentiment_label, sentiment_score
                        else:
                            # 如果无法解析JSON，尝试从文本中提取情感信息
                            if "积极" in content:
                                return "积极", 0.7
                            elif "消极" in content or "负面" in content:
                                return "消极", -0.7
                            else:
                                return "中性", 0.0
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，尝试从文本中提取情感信息
                        if "积极" in content:
                            return "积极", 0.7
                        elif "消极" in content or "负面" in content:
                            return "消极", -0.7
                        else:
                            return "中性", 0.0
                else:
                    print(f"API响应格式异常: {result}")
                    if attempt < max_retries - 1:
                        print(f"尝试 {attempt + 1}/{max_retries} 失败，重试中...")
                        time.sleep(2)
                        continue
                    else:
                        return "中性", 0.0
            else:
                print(f"API请求失败: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"尝试 {attempt + 1}/{max_retries} 失败，重试中...")
                    time.sleep(2)
                    continue
                else:
                    return "中性", 0.0
                    
        except Exception as e:
            print(f"分析异常: {str(e)}")
            if attempt < max_retries - 1:
                print(f"尝试 {attempt + 1}/{max_retries} 失败，重试中...")
                time.sleep(2)
                continue
            else:
                return "中性", 0.0
    
    return "中性", 0.0

def batch_analyze_sentiments(data_file, access_token):
    """批量分析情感"""
    # 读取数据
    print(f"读取数据文件: {data_file}")
    df = pd.read_csv(data_file)
    print(f"原始数据包含 {len(df)} 条评论")
    
    # 检查是否已有分析结果
    if 'llm_sentiment_label_new' in df.columns and 'llm_sentiment_score_new' in df.columns:
        print("检测到已有情感分析结果，将跳过已分析的评论")
        # 找出未分析的行
        mask = df['llm_sentiment_label_new'].isna() | df['llm_sentiment_score_new'].isna()
        comments_to_analyze = df[mask]
        print(f"需要分析 {len(comments_to_analyze)} 条新评论")
    else:
        comments_to_analyze = df
        print(f"需要分析 {len(comments_to_analyze)} 条评论")
    
    # 分析每条评论
    results = []
    total = len(comments_to_analyze)
    
    for i, row in comments_to_analyze.iterrows():
        text = row.get('post_title', '') or row.get('combined_text', '') or row.get('processed_content', '')
        
        print(f"分析第 {i+1}/{total} 条评论: {text[:50]}...")
        
        sentiment_label, sentiment_score = analyze_sentiment_with_qianfan(text, access_token)
        
        results.append({
            'index': i,
            'llm_sentiment_label_new': sentiment_label,
            'llm_sentiment_score_new': sentiment_score
        })
        
        # 每20条保存一次中间结果
        if (i + 1) % 20 == 0:
            print(f"已处理 {i+1} 条评论，保存中间结果...")
            temp_df = df.copy()
            for result in results:
                idx = result['index']
                temp_df.at[idx, 'llm_sentiment_label_new'] = result['llm_sentiment_label_new']
                temp_df.at[idx, 'llm_sentiment_score_new'] = result['llm_sentiment_score_new']
            
            # 计算融合分数
            if 'lexicon_sentiment' in temp_df.columns:
                temp_df['ensemble_sentiment_score_new'] = (
                    temp_df['lexicon_sentiment'] * 0.3 + 
                    temp_df['llm_sentiment_score_new'] * 0.7
                )
            else:
                temp_df['ensemble_sentiment_score_new'] = temp_df['llm_sentiment_score_new']
            
            temp_df.to_csv(f"temp_qianfan_sentiment_analysis.csv", index=False, encoding='utf-8-sig')
    
    # 更新原始DataFrame
    for result in results:
        idx = result['index']
        df.at[idx, 'llm_sentiment_label_new'] = result['llm_sentiment_label_new']
        df.at[idx, 'llm_sentiment_score_new'] = result['llm_sentiment_score_new']
    
    # 计算融合分数
    if 'lexicon_sentiment' in df.columns:
        df['ensemble_sentiment_score_new'] = (
            df['lexicon_sentiment'] * 0.3 + 
            df['llm_sentiment_score_new'] * 0.7
        )
    else:
        df['ensemble_sentiment_score_new'] = df['llm_sentiment_score_new']
    
    # 保存结果
    output_file = data_file.replace('.csv', '_qianfan_sentiment_analysis.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存百度千帆情感分析结果到: {output_file}")
    
    # 统计结果
    sentiment_counts = df['llm_sentiment_label_new'].value_counts()
    print("\n=== 情感分析统计 ===")
    print(f"总评论数: {len(df)}")
    for label, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        print(f"{label}评论: {count} ({percentage:.1f}%)")
    
    # 统计分数
    if 'llm_sentiment_score_new' in df.columns:
        score_mean = df['llm_sentiment_score_new'].mean()
        score_std = df['llm_sentiment_score_new'].std()
        print(f"\n情感分数统计:")
        print(f"百度千帆LLM分数 - 均值: {score_mean:.4f}, 标准差: {score_std:.4f}")
    
    return df

def main():
    """主函数"""
    print("百度千帆大模型情感分析工具")
    print("=" * 50)
    
    # 检查SECRET_KEY
    if SECRET_KEY == "请在这里提供SECRET_KEY":
        print("错误: 请在代码中提供百度千帆的SECRET_KEY")
        print("您可以在百度千帆控制台获取API Key和Secret Key")
        print("获取方式: 登录百度智能云 -> 千帆大模型平台 -> 应用接入 -> 创建应用 -> 获取API Key和Secret Key")
        return
    
    # 获取access token
    print("获取百度千帆access token...")
    access_token = get_access_token(API_KEY, SECRET_KEY)
    
    if not access_token:
        print("错误: 无法获取access token，请检查API Key和Secret Key是否正确")
        return
    
    print(f"成功获取access token: {access_token[:20]}...")
    
    # 分析情感
    data_file = "300059_sentiment_analysis.csv"
    if os.path.exists(data_file):
        result_df = batch_analyze_sentiments(data_file, access_token)
        print("\n情感分析完成！")
        print(f"请查看 {data_file.replace('.csv', '_qianfan_sentiment_analysis.csv')} 文件获取结果。")
    else:
        print(f"错误: 找不到数据文件 {data_file}")

if __name__ == "__main__":
    main()
