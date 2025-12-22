import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import time
import re
import os
from datetime import datetime

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载免费开源大模型（自动下载，无需手动配置）
print("正在加载Qwen-1.8B-Chat模型...")
model_name = "Qwen/Qwen-1.8B-Chat"  # 轻量模型，4G内存即可运行

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    print("尝试使用更小的模型...")
    try:
        # 如果Qwen-1.8B加载失败，尝试更小的模型
        model_name = "Qwen/Qwen-1.8B-Chat"  # 使用相同的模型但可能需要不同的参数
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 使用float32以兼容更多设备
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("使用float32精度加载模型成功！")
    except Exception as e2:
        print(f"模型加载仍然失败: {str(e2)}")
        print("请检查网络连接或尝试其他模型")
        exit(1)

def llm_sentiment_analysis(text, max_retries=2):
    """
    使用本地大模型分析文本情感
    
    参数:
    - text: 要分析的文本
    - max_retries: 最大重试次数
    
    返回:
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感分数（-1到1之间）
    """
    if pd.isna(text) or text.strip() == "":
        return "中性", 0.0
    
    # 限制文本长度，避免超出模型限制
    text = text.strip()[:500]
    
    # 构建提示词
    prompt = f"""请分析以下股吧评论的情感倾向，返回JSON格式的结果：

文本内容：{text}

请按照以下格式返回：
{{
    "sentiment_label": "积极/中性/消极",
    "sentiment_score": 数值（-1到1之间，积极为正，消极为负，中性为0）
}}

请只返回JSON，不要添加其他解释。"""
    
    for attempt in range(max_retries):
        try:
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 提取模型生成的部分（去掉输入的prompt）
            if prompt in result:
                result = result.replace(prompt, "").strip()
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    import json
                    sentiment_data = json.loads(json_str)
                    
                    sentiment_label = sentiment_data.get("sentiment_label", "中性")
                    sentiment_score = float(sentiment_data.get("sentiment_score", 0))
                    
                    # 确保分数在-1到1之间
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                    
                    return sentiment_label, sentiment_score
                else:
                    # 如果无法解析JSON，尝试从文本中提取情感信息
                    if "积极" in result:
                        return "积极", 0.7
                    elif "消极" in result or "负面" in result:
                        return "消极", -0.7
                    else:
                        return "中性", 0.0
            except:
                # 如果JSON解析失败，尝试从文本中提取情感信息
                if "积极" in result:
                    return "积极", 0.7
                elif "消极" in result or "负面" in result:
                    return "消极", -0.7
                else:
                    return "中性", 0.0
                    
        except Exception as e:
            print(f"分析异常: {str(e)}")
            if attempt < max_retries - 1:
                print(f"尝试 {attempt + 1}/{max_retries} 失败，重试中...")
                time.sleep(1)
                continue
            else:
                return "中性", 0.0
    
    return "中性", 0.0

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
        
        sentiment_label, sentiment_score = llm_sentiment_analysis(text)
        
        results.append({
            'index': i,
            'llm_sentiment_label_new': sentiment_label,
            'llm_sentiment_score_new': sentiment_score
        })
        
        # 每10条保存一次中间结果（本地模型处理较慢）
        if (i + 1) % 10 == 0:
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
            
            temp_df.to_csv(f"temp_local_llm_sentiment_analysis.csv", index=False, encoding='utf-8-sig')
            
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
    output_file = data_file.replace('.csv', '_local_llm_sentiment_analysis.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存本地大模型情感分析结果到: {output_file}")
    
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
        print(f"本地LLM分数 - 均值: {score_mean:.4f}, 标准差: {score_std:.4f}")
    
    # 计算总用时
    total_time = time.time() - start_time
    print(f"\n总用时: {total_time:.1f}秒, 平均每条评论: {total_time/len(df):.2f}秒")
    
    return df

def main():
    """主函数"""
    print("本地大模型情感分析工具")
    print("=" * 50)
    
    # 分析情感
    data_file = "300059_sentiment_analysis.csv"
    if os.path.exists(data_file):
        result_df = batch_analyze_sentiments(data_file)
        print("\n情感分析完成！")
        print(f"请查看 {data_file.replace('.csv', '_local_llm_sentiment_analysis.csv')} 文件获取结果。")
    else:
        print(f"错误: 找不到数据文件 {data_file}")

if __name__ == "__main__":
    main()
