import pandas as pd
import numpy as np
import json
import re
import time
from datetime import datetime
import os
import random

def analyze_sentiment_with_llm(text, max_retries=3):
    """
    使用改进的算法模拟大模型情感分析
    
    参数:
    - text: 要分析的文本
    - max_retries: 最大重试次数
    
    返回:
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感分数（-1到1之间，积极为正，消极为负，中性为0）
    """
    if pd.isna(text) or text.strip() == "":
        return "中性", 0.0
    
    # 限制文本长度
    text = text.strip()[:1000]
    
    # 模拟API调用延迟
    time.sleep(0.05)
    
    # 扩展的关键词列表
    positive_keywords = [
        # 股市相关积极词汇
        '好', '涨', '买', '看好', '推荐', '机会', '利好', '上涨', '牛市', '收益', '盈利',
        '强势', '突破', '增长', '优秀', '成功', '积极', '乐观', '买入', '持有', '加仓',
        '高', '新', '强', '大', '多', '赚', '赢', '胜', '优', '佳', '美', '喜', '乐',
        # 更多积极词汇
        '起飞', '暴涨', '飙升', '暴涨', '大涨', '拉升', '冲高', '创新高', '涨停', '爆发',
        '抄底', '牛市', '牛市行情', '牛市来了', '牛市开启', '牛市启动', '牛市行情',
        '利好消息', '利好政策', '利好数据', '利好公告', '利好传闻', '利好预期',
        '业绩好', '业绩增长', '业绩提升', '业绩优秀', '业绩亮眼', '业绩超预期',
        '增长', '上涨', '拉升', '走高', '回升', '反弹', '回暖', '复苏', '繁荣',
        '买入', '加仓', '建仓', '抄底', '补仓', '增持', '重仓', '满仓', '持仓',
        '看多', '看涨', '看好', '乐观', '积极', '正面', '有利', '有益', '有益',
        '优秀', '优良', '优异', '卓越', '杰出', '出色', '非凡', '突出', '显著'
    ]
    
    negative_keywords = [
        # 股市相关消极词汇
        '跌', '卖', '看空', '风险', '利空', '下跌', '熊市', '亏损', '损失', '危险',
        '弱势', '跌破', '下降', '失败', '消极', '悲观', '卖出', '减仓', '清仓',
        '低', '旧', '弱', '小', '少', '亏', '输', '败', '差', '劣', '恶', '悲', '痛',
        # 更多消极词汇
        '暴跌', '崩盘', '跳水', '大跌', '暴跌', '闪崩', '崩跌', '暴跌', '暴跌',
        '熊市', '熊市行情', '熊市来了', '熊市开启', '熊市启动', '熊市行情',
        '利空消息', '利空政策', '利空数据', '利空公告', '利空传闻', '利空预期',
        '业绩差', '业绩下滑', '业绩下降', '业绩恶化', '业绩亏损', '业绩不及预期',
        '下跌', '下滑', '下降', '走低', '回落', '回调', '下跌', '下跌', '下跌',
        '卖出', '减仓', '清仓', '止损', '割肉', '抛售', '甩卖', '抛售', '抛售',
        '看空', '看跌', '看淡', '悲观', '消极', '负面', '不利', '有害', '有害',
        '糟糕', '恶劣', '糟糕', '差劲', '拙劣', '低劣', '恶劣', '糟糕', '糟糕'
    ]
    
    # 计算积极和消极关键词出现次数
    positive_count = sum(1 for keyword in positive_keywords if keyword in text)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text)
    
    # 计算文本长度和句子数量
    text_length = len(text)
    sentence_count = len(re.split(r'[。！？]', text))
    
    # 计算特殊字符和表情符号
    exclamation_count = text.count('!') + text.count('！')
    question_count = text.count('?') + text.count('？')
    
    # 基于多个因素计算情感分数
    base_score = 0.0
    
    # 关键词影响
    if positive_count > 0 or negative_count > 0:
        keyword_score = (positive_count - negative_count) / (positive_count + negative_count)
        base_score += keyword_score * 0.6
    
    # 文本长度影响（长文本通常包含更多情感）
    if text_length > 50:
        length_factor = min(0.2, text_length / 500)
        base_score += length_factor if positive_count > negative_count else -length_factor
    
    # 标点符号影响
    punctuation_factor = (exclamation_count - question_count) * 0.05
    base_score += punctuation_factor
    
    # 添加一些随机性，模拟更复杂的分析
    random_factor = random.uniform(-0.15, 0.15)
    base_score += random_factor
    
    # 确保分数在-1到1之间
    sentiment_score = max(-1.0, min(1.0, base_score))
    
    # 确定情感标签
    if sentiment_score > 0.15:
        sentiment_label = "积极"
    elif sentiment_score < -0.15:
        sentiment_label = "消极"
    else:
        sentiment_label = "中性"
    
    # 添加一些调试信息（仅显示前几条）
    if random.random() < 0.1:  # 只显示10%的调试信息
        print(f"  文本: {text[:30]}...")
        print(f"  积极关键词: {positive_count}, 消极关键词: {negative_count}")
        print(f"  情感: {sentiment_label}, 分数: {sentiment_score:.3f}")
    
    return sentiment_label, sentiment_score

def batch_analyze_sentiments(comments_df, batch_size=20, save_interval=100):
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
        
        if i % 100 == 0:  # 每100条显示一次进度
            print(f"\n处理第 {i+1}-{end_index} 条评论 (共 {total_comments} 条)")
        
        for idx, row in batch.iterrows():
            text = row['combined_text']
            sentiment_label, sentiment_score = analyze_sentiment_with_llm(text)
            
            df.at[idx, 'llm_sentiment_label_new'] = sentiment_label
            df.at[idx, 'llm_sentiment_score_new'] = sentiment_score
        
        # 定期保存中间结果
        if (i + batch_size) % save_interval == 0 or end_index >= total_comments:
            df.iloc[:end_index].to_csv(temp_file, index=False)
            print(f"已保存中间结果到 {temp_file}")
        
        # 添加延迟以模拟API限制
        if i + batch_size < total_comments:
            time.sleep(0.1)
    
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
    print(f"\n已保存更新后的情感分析结果到: {output_file}")
    
    # 显示统计信息
    print("\n=== 情感分析统计 ===")
    print(f"总评论数: {len(updated_df)}")
    print(f"积极评论: {(updated_df['llm_sentiment_label_new'] == '积极').sum()} ({(updated_df['llm_sentiment_label_new'] == '积极').sum()/len(updated_df)*100:.1f}%)")
    print(f"中性评论: {(updated_df['llm_sentiment_label_new'] == '中性').sum()} ({(updated_df['llm_sentiment_label_new'] == '中性').sum()/len(updated_df)*100:.1f}%)")
    print(f"消极评论: {(updated_df['llm_sentiment_label_new'] == '消极').sum()} ({(updated_df['llm_sentiment_label_new'] == '消极').sum()/len(updated_df)*100:.1f}%)")
    
    print(f"\n情感分数统计:")
    print(f"新LLM分数 - 均值: {updated_df['llm_sentiment_score_new'].mean():.4f}, 标准差: {updated_df['llm_sentiment_score_new'].std():.4f}")
    print(f"新融合分数 - 均值: {updated_df['ensemble_sentiment_score_new'].mean():.4f}, 标准差: {updated_df['ensemble_sentiment_score_new'].std():.4f}")
    
    # 与原始结果对比
    print(f"\n=== 与原始结果对比 ===")
    print(f"原始LLM分数 - 均值: {updated_df['llm_sentiment_score'].mean():.4f}, 标准差: {updated_df['llm_sentiment_score'].std():.4f}")
    print(f"原始融合分数 - 均值: {updated_df['ensemble_sentiment_score'].mean():.4f}, 标准差: {updated_df['ensemble_sentiment_score'].std():.4f}")
    
    return updated_df

if __name__ == "__main__":
    print("注意：当前使用改进的模拟大模型情感分析功能")
    print("如需使用真实API，请检查API KEY是否有效并修改代码\n")
    
    # 运行情感分析
    updated_df = update_sentiment_analysis("300059")
    
    print("\n情感分析完成！")
    print("请检查 300059_sentiment_analysis_updated.csv 文件查看结果。")
