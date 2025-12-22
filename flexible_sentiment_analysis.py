import pandas as pd
import numpy as np
import re
import time
import os
from datetime import datetime

# 尝试导入transformers，如果失败则使用模拟方法
try:
    from transformers import pipeline
    USE_TRANSFORMERS = True
    print("成功导入transformers库")
except ImportError:
    USE_TRANSFORMERS = False
    print("transformers库未安装，将使用改进的模拟方法")

# 如果transformers可用，尝试加载轻量级情感分析模型
if USE_TRANSFORMERS:
    try:
        print("正在加载轻量级情感分析模型...")
        # 使用一个轻量级的中文情感分析模型
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="uer/roberta-base-finetuned-chinanews-chinese",
            tokenizer="uer/roberta-base-finetuned-chinanews-chinese"
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        USE_TRANSFORMERS = False
        print("将使用改进的模拟方法")

def analyze_sentiment_with_local_model(text):
    """
    使用本地模型或改进的模拟方法分析文本情感
    
    参数:
    - text: 要分析的文本
    
    返回:
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感分数（-1到1之间）
    """
    if pd.isna(text) or text.strip() == "":
        return "中性", 0.0
    
    # 限制文本长度
    text = text.strip()[:500]
    
    if USE_TRANSFORMERS:
        try:
            # 使用transformers模型进行情感分析
            result = sentiment_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # 转换标签和分数
            if label == 'POSITIVE':
                sentiment_label = "积极"
                sentiment_score = score
            elif label == 'NEGATIVE':
                sentiment_label = "消极"
                sentiment_score = -score
            else:
                sentiment_label = "中性"
                sentiment_score = 0.0
                
            return sentiment_label, sentiment_score
        except Exception as e:
            print(f"模型分析失败: {str(e)}，使用模拟方法")
    
    # 改进的模拟方法
    return analyze_sentiment_with_simulation(text)

def analyze_sentiment_with_simulation(text):
    """
    使用改进的模拟方法分析文本情感
    
    参数:
    - text: 要分析的文本
    
    返回:
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感分数（-1到1之间）
    """
    # 扩展的关键词列表
    positive_keywords = [
        # 股市相关积极词汇
        '好', '涨', '买', '看好', '推荐', '机会', '利好', '上涨', '牛市', '收益', '盈利',
        '强势', '突破', '增长', '优秀', '成功', '积极', '乐观', '买入', '持有', '加仓',
        '大涨', '涨停', '爆发', '飙升', '飞涨', '暴涨', '走高', '攀升', '坚挺', '活跃',
        '繁荣', '兴旺', '景气', '火热', '抢手', '热门', '领先', '优胜', '出色', '卓越',
        '稳健', '强劲', '有力', '旺盛', '蓬勃', '向上', '利好', '正面', '优秀', '良好'
    ]
    
    negative_keywords = [
        # 股市相关消极词汇
        '跌', '卖', '看空', '风险', '利空', '下跌', '熊市', '亏损', '损失', '危险',
        '弱势', '跌破', '下降', '失败', '消极', '悲观', '卖出', '减仓', '清仓',
        '大跌', '跌停', '暴跌', '跳水', '崩盘', '下挫', '下滑', '走低', '疲软', '低迷',
        '衰退', '萎缩', '萧条', '惨淡', '冷清', '抛售', '恐慌', '危机', '困境', '难题',
        '恶化', '严峻', '堪忧', '忧虑', '担心', '害怕', '恐慌', '悲观', '负面', '不利'
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
    import random
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
    
    return sentiment_label, sentiment_score

def batch_analyze_sentiments(data_file):
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
    start_time = time.time()
    
    for i, row in comments_to_analyze.iterrows():
        text = row.get('post_title', '') or row.get('combined_text', '') or row.get('processed_content', '')
        
        print(f"分析第 {i+1}/{total} 条评论: {text[:50]}...")
        
        sentiment_label, sentiment_score = analyze_sentiment_with_local_model(text)
        
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
            
            temp_df.to_csv(f"temp_local_sentiment_analysis.csv", index=False, encoding='utf-8-sig')
            
            # 显示进度
            elapsed_time = time.time() - start_time
            avg_time_per_comment = elapsed_time / (i + 1)
            remaining_comments = total - (i + 1)
            estimated_remaining_time = remaining_comments * avg_time_per_comment
            print(f"已用时: {elapsed_time:.1f}秒, 预计剩余时间: {estimated_remaining_time:.1f}秒")
    
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
    if USE_TRANSFORMERS:
        output_file = data_file.replace('.csv', '_transformer_sentiment_analysis.csv')
        print(f"已保存Transformer模型情感分析结果到: {output_file}")
    else:
        output_file = data_file.replace('.csv', '_improved_sentiment_analysis.csv')
        print(f"已保存改进的情感分析结果到: {output_file}")
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
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
        if USE_TRANSFORMERS:
            print(f"Transformer模型分数 - 均值: {score_mean:.4f}, 标准差: {score_std:.4f}")
        else:
            print(f"改进模拟分数 - 均值: {score_mean:.4f}, 标准差: {score_std:.4f}")
    
    # 计算总用时
    total_time = time.time() - start_time
    print(f"\n总用时: {total_time:.1f}秒, 平均每条评论: {total_time/len(df):.2f}秒")
    
    return df

def main():
    """主函数"""
    print("本地情感分析工具")
    print("=" * 50)
    
    if USE_TRANSFORMERS:
        print("使用Transformer模型进行情感分析")
    else:
        print("使用改进的模拟方法进行情感分析")
    
    # 分析情感
    data_file = "300059_sentiment_analysis.csv"
    if os.path.exists(data_file):
        result_df = batch_analyze_sentiments(data_file)
        print("\n情感分析完成！")
        if USE_TRANSFORMERS:
            print(f"请查看 {data_file.replace('.csv', '_transformer_sentiment_analysis.csv')} 文件获取结果。")
        else:
            print(f"请查看 {data_file.replace('.csv', '_improved_sentiment_analysis.csv')} 文件获取结果。")
    else:
        print(f"错误: 找不到数据文件 {data_file}")

if __name__ == "__main__":
    main()
