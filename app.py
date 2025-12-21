import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor
import warnings
import os
import requests
from collections import Counter
import re
import matplotlib.font_manager as fm

# 设置中文显示 - 修复云端中文乱码问题
def setup_chinese_font():
    """设置中文字体，解决云端环境中文显示问题"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 如果设置失败，使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="东方财富股吧评论情感分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载情感词典 - 移除缓存以确保参数更新生效
def load_sentiment_dictionaries():
    """
    加载用户提供的情感词典
    积极词典：zhang_unformal_pos (1).txt
    消极词典：zhang_unformal_neg (1).txt
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建词典文件路径
    pos_dict_path = os.path.join(script_dir, 'zhang_unformal_pos (1).txt')
    neg_dict_path = os.path.join(script_dir, 'zhang_unformal_neg (1).txt')
    
    # 检查文件是否存在
    if not os.path.exists(pos_dict_path) or not os.path.exists(neg_dict_path):
        st.error(f"情感词典文件未找到，请确保以下文件存在于应用目录中：\n- {pos_dict_path}\n- {neg_dict_path}")
        st.stop()
    
    # 加载积极词典
    with open(pos_dict_path, 'r', encoding='utf-8') as f:
        positive_words = [line.strip() for line in f if line.strip()]
    
    # 加载消极词典
    with open(neg_dict_path, 'r', encoding='utf-8') as f:
        negative_words = [line.strip() for line in f if line.strip()]
    
    return positive_words, negative_words

# 实现基于词典的情感分析
def lexicon_based_sentiment_analysis(text, pos_words, neg_words):
    """
    基于词典的情感分析
    text: 评论文本
    pos_words: 积极词语列表
    neg_words: 消极词语列表
    返回：
    - sentiment_label: 情感标签（积极/中性/消极）
    - sentiment_score: 情感得分（-1到1之间）
    """
    if pd.isna(text) or text.strip() == '':
        return '中性', 0.0
    
    # 计算积极词语和消极词语的出现次数
    pos_count = sum(1 for word in pos_words if word in text)
    neg_count = sum(1 for word in neg_words if word in text)
    
    # 计算情感得分
    total = pos_count + neg_count + 1  # 加1避免除以0
    sentiment_score = (pos_count - neg_count) / total
    
    # 确定情感标签
    if sentiment_score > 0.1:
        sentiment_label = '积极'
    elif sentiment_score < -0.1:
        sentiment_label = '消极'
    else:
        sentiment_label = '中性'
    
    return sentiment_label, sentiment_score

# 设置页面标题
st.title('东方财富股吧评论情感分析')

# 加载数据 - 移除缓存以确保参数更新生效
def load_data():
    # 加载评论和情感分析数据
    # 优先使用更新后的情感分析结果文件
    updated_file = "300059_sentiment_analysis_updated.csv"
    original_file = "300059_sentiment_analysis.csv"
    
    try:
        if os.path.exists(updated_file):
            comments_df = pd.read_csv(updated_file)
            st.success(f"已加载改进的情感分析结果（{len(comments_df)}条评论）")
        elif os.path.exists(original_file):
            comments_df = pd.read_csv(original_file)
            st.info(f"已加载原始情感分析结果（{len(comments_df)}条评论）")
        else:
            st.error("未找到情感分析数据文件，请确保以下文件之一存在于应用目录中：\n- 300059_sentiment_analysis_updated.csv\n- 300059_sentiment_analysis.csv")
            st.stop()
        
        comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
        
        # 加载价格数据
        price_file = "300059_price_data.csv"
        if os.path.exists(price_file):
            price_df = pd.read_csv(price_file)
            price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        else:
            st.error("未找到价格数据文件，请确保以下文件存在于应用目录中：\n- 300059_price_data.csv")
            st.stop()
        
        return comments_df, price_df
    except Exception as e:
        st.error(f"加载数据时发生错误：{str(e)}")
        st.stop()

# 处理数据 - 移除缓存以确保参数更新生效
def process_data(comments_df, price_df, text_length_limit=500, window_size=30, lag_days=0):
    # 处理combined_text字段为空的情况
    filtered_comments = comments_df.copy()
    
    # 调整文本字段优先级：优先使用post_title，再使用combined_text和processed_content
    if 'post_title' in filtered_comments.columns:
        filtered_comments['combined_text'] = filtered_comments['post_title']
    elif 'combined_text' in filtered_comments.columns:
        filtered_comments['combined_text'] = filtered_comments['combined_text']
    else:
        st.error("数据中未找到有效的文本列（post_title或combined_text）")
        return pd.DataFrame(), pd.DataFrame()
    
    # 过滤无效评论内容
    invalid_pattern = r'(图片图片|转发转发|^[!！]{5,}$|^[?？]{5,}$|^\.{5,}$|^\s*$)'
    filtered_comments = filtered_comments[~filtered_comments['combined_text'].str.contains(invalid_pattern, na=False, regex=True)]
    
    # 加载情感词典
    try:
        positive_words, negative_words = load_sentiment_dictionaries()
    except Exception as e:
        st.error(f"加载情感词典失败：{str(e)}")
        return pd.DataFrame(), pd.DataFrame()
    
    # 应用基于词典的情感分析
    sentiment_results = filtered_comments['combined_text'].apply(
        lambda x: lexicon_based_sentiment_analysis(x, positive_words, negative_words)
    )
    
    # 将结果拆分为情感标签和情感得分
    filtered_comments['llm_sentiment_label'] = sentiment_results.str[0]
    filtered_comments['llm_sentiment_score'] = sentiment_results.str[1]
    filtered_comments['ensemble_sentiment_score'] = sentiment_results.str[1]
    filtered_comments['lexicon_sentiment'] = sentiment_results.str[1]
    
    # 文本长度过滤
    filtered_comments['text_length'] = filtered_comments['combined_text'].str.len()
    filtered_comments = filtered_comments[(filtered_comments['text_length'] >= 1) & (filtered_comments['text_length'] <= text_length_limit)]
    
    # 按日期聚合情感数据
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg({
        'ensemble_sentiment_score': ['mean', 'median', 'std', 'count'],
        'llm_sentiment_score': 'mean',
        'lexicon_sentiment': 'mean'
    }).reset_index()
    
    # 重命名列
    daily_sentiment.columns = ['date', 'ensemble_mean', 'ensemble_median', 'ensemble_std', 'comment_count', 'llm_mean', 'lexicon_mean']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 合并价格数据（使用左连接，保留所有价格日期）
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    
    # 处理没有评论的日期（填充NaN值）
    merged_df['comment_count'] = merged_df['comment_count'].fillna(0)
    merged_df['ensemble_mean'] = merged_df['ensemble_mean'].fillna(0)
    merged_df['ensemble_median'] = merged_df['ensemble_median'].fillna(0)
    merged_df['ensemble_std'] = merged_df['ensemble_std'].fillna(0)
    merged_df['llm_mean'] = merged_df['llm_mean'].fillna(0)
    merged_df['lexicon_mean'] = merged_df['lexicon_mean'].fillna(0)
    
    # 确保std列不为NaN
    merged_df['ensemble_std'] = merged_df['ensemble_std'].fillna(0)
    
    # 添加滞后情感数据
    if lag_days > 0:
        merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean'].shift(lag_days)
        merged_df['comment_count_lag'] = merged_df['comment_count'].shift(lag_days)
        merged_df['ensemble_std_lag'] = merged_df['ensemble_std'].shift(lag_days)
        merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean_lag'].fillna(0)
        merged_df['comment_count_lag'] = merged_df['comment_count_lag'].fillna(0)
        merged_df['ensemble_std_lag'] = merged_df['ensemble_std_lag'].fillna(0)
    
    # 计算移动平均
    if window_size > 1:
        merged_df['ensemble_mean_rolling'] = merged_df['ensemble_mean'].rolling(window=window_size).mean()
        merged_df['next_day_return_rolling'] = merged_df['next_day_return'].rolling(window=window_size).mean()
    
    return merged_df, filtered_comments

# 侧边栏：参数调整
st.sidebar.subheader('参数调整')

# 使用session_state管理参数状态
if 'text_length' not in st.session_state:
    st.session_state.text_length = 500
if 'window_size' not in st.session_state:
    st.session_state.window_size = 30
if 'lag_days' not in st.session_state:
    st.session_state.lag_days = 0

# 重置按钮
if st.sidebar.button('🔄 重置所有参数'):
    st.session_state.text_length = 500
    st.session_state.window_size = 30
    st.session_state.lag_days = 0
    st.experimental_rerun()

text_length = st.sidebar.slider('文本长度限制', 50, 1000, st.session_state.text_length, step=50, key='length_slider')
window_size = st.sidebar.slider('移动平均窗口大小(天)', 1, 90, st.session_state.window_size, step=5, key='window_slider')
lag_days = st.sidebar.slider('情感滞后天数', 0, 10, st.session_state.lag_days, step=1, key='lag_slider')

# 更新session_state
st.session_state.text_length = text_length
st.session_state.window_size = window_size
st.session_state.lag_days = lag_days

# 加载和处理数据
try:
    comments_df, price_df = load_data()
    merged_df, filtered_comments = process_data(comments_df, price_df, text_length, window_size, lag_days)
    
    # 数据质量检查
    st.subheader('数据质量检查')
    
    # 检查评论数量
    total_comments = len(comments_df)
    filtered_count = len(filtered_comments)
    filtered_out_count = total_comments - filtered_count
    zero_sentiment = (comments_df['ensemble_sentiment_score'] == 0).sum()
    
    st.write(f'📊 数据概览：')
    st.write(f'- 共收集到 {total_comments} 条评论')
    st.write(f'- 经过过滤后保留：{filtered_count} 条有效评论')
    st.write(f'- 过滤掉的评论：{filtered_out_count} 条（内容无效或不符合长度要求）')
    st.write(f'- 中性情感评论（分数为0）：{zero_sentiment} 条')
    st.write(f'- 保留的交易日数据：{len(merged_df)} 个')
    
    # 显示警告信息
    if filtered_count < total_comments * 0.5:
        st.warning(f'注意：有 {filtered_out_count} 条评论被过滤，保留的有效样本较少，可能影响分析结果的准确性。')
    
    if zero_sentiment > total_comments * 0.8:
        st.warning(f'注意：{zero_sentiment/total_comments:.1%} 的评论情感分数为0，可能影响分析结果的准确性。')
    
    # 检查日期范围
    if not merged_df.empty:
        date_range = f'{merged_df["trade_date"].min().strftime("%Y-%m-%d")} 至 {merged_df["trade_date"].max().strftime("%Y-%m-%d")}'
        st.write(f'- 数据日期范围：{date_range}')
    
    # 显示评论数量随时间的变化
    st.subheader('评论数量随时间变化')
    
    try:
        # 按日期分组并计算评论数量
        daily_comments = comments_df.groupby(comments_df['post_publish_time'].dt.date)['post_id'].count()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 检查是否有数据
        if len(daily_comments) > 0:
            # 绘制折线图
            daily_comments.plot(ax=ax, marker='o', linestyle='-', linewidth=2, markersize=5, color='#1f77b4')
            
            # 添加每日评论数量标签
            for x, y in zip(daily_comments.index, daily_comments.values):
                ax.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9)
            
            # 设置图表标题和标签
            ax.set_title('每日评论数量变化趋势', fontsize=14)
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('评论数量', fontsize=12)
            
            # 调整Y轴范围，确保所有点都能显示
            ax.set_ylim(0, daily_comments.max() * 1.1)
            
            # 添加网格线
            ax.grid(True, alpha=0.3)
            
            # 调整日期标签
            plt.xticks(rotation=45, fontsize=10)
            
            # 计算统计信息
            avg_daily = daily_comments.mean()
            max_daily = daily_comments.max()
            min_daily = daily_comments.min()
            
            # 在图表中添加统计信息
            stats_text = f'平均日评论数: {avg_daily:.1f}\n最高日评论数: {max_daily}\n最低日评论数: {min_daily}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        else:
            ax.set_title('暂无评论数据', fontsize=14)
            ax.text(0.5, 0.5, '没有足够的评论数据来绘制趋势图', transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表
        st.pyplot(fig)
        
        # 显示统计信息
        if len(daily_comments) > 0:
            st.write(f'📊 评论数量统计：')
            st.write(f'- 评论日期范围：{daily_comments.index.min()} 至 {daily_comments.index.max()}')
            st.write(f'- 有评论的天数：{len(daily_comments)} 天')
            st.write(f'- 平均每日评论数：{daily_comments.mean():.1f} 条')
            st.write(f'- 最高每日评论数：{daily_comments.max()} 条')
            st.write(f'- 最低每日评论数：{daily_comments.min()} 条')
        else:
            st.warning('没有评论数据可显示。')
    except Exception as e:
        st.error(f'绘制评论数量趋势图时发生错误：{str(e)}')
        st.write('请检查数据格式或尝试调整参数。')
    
    # 显示情感分析结果
    st.subheader('情感分析结果')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # LLM情感标签分布
        st.write('### 情感标签分布')
        try:
            # 检查情感标签列是否存在且非空
            if 'llm_sentiment_label' in comments_df.columns:
                sentiment_counts = comments_df['llm_sentiment_label'].value_counts()
                
                if len(sentiment_counts) > 0:
                    # 创建饼图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # 设置饼图颜色
                    colors = ['#4caf50' if label == '积极' else '#ff9800' if label == '中性' else '#f44336' for label in sentiment_counts.index]
                    
                    # 绘制饼图
                    patches, texts, autotexts = ax.pie(
                        sentiment_counts.values, 
                        labels=sentiment_counts.index, 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=colors, 
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}, 
                        textprops={'fontsize': 12}
                    )
                    
                    # 设置百分比标签颜色和大小
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(11)
                    
                    # 设置标题
                    ax.set_title('LLM情感标签分布', fontsize=14)
                    
                    # 确保饼图是圆形
                    ax.axis('equal')
                    
                    # 显示图表
                    st.pyplot(fig)
                    
                    # 显示具体数量
                    st.write('情感标签数量：')
                    for label, count in sentiment_counts.items():
                        st.write(f'- {label}: {count} 条 ({count/len(comments_df)*100:.1f}%)')
                else:
                    st.write('暂无情感标签数据')
            else:
                st.write('数据中没有情感标签列')
        except Exception as e:
            st.error(f'绘制情感标签分布图时发生错误：{str(e)}')
    
    with col2:
        # 融合情感得分分布
        st.write('### 情感得分分布')
        try:
            # 检查情感得分列是否存在
            if 'ensemble_sentiment_score' in comments_df.columns:
                # 计算统计信息
                mean_score = comments_df['ensemble_sentiment_score'].mean()
                median_score = comments_df['ensemble_sentiment_score'].median()
                std_score = comments_df['ensemble_sentiment_score'].std()
                
                # 创建直方图
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # 绘制直方图
                sns.histplot(
                    comments_df['ensemble_sentiment_score'], 
                    bins=20, 
                    kde=True, 
                    ax=ax, 
                    color='#1f77b4', 
                    edgecolor='w'
                )
                
                # 添加均值和中位数线
                ax.axvline(mean_score, color='red', linestyle='--', label=f'均值: {mean_score:.2f}')
                ax.axvline(median_score, color='green', linestyle='--', label=f'中位数: {median_score:.2f}')
                
                # 设置图表标题和标签
                ax.set_title('融合情感得分分布', fontsize=14)
                ax.set_xlabel('情感得分', fontsize=12)
                ax.set_ylabel('评论数量', fontsize=12)
                
                # 添加网格线
                ax.grid(True, alpha=0.3)
                
                # 添加图例
                ax.legend()
                
                # 调整布局
                plt.tight_layout()
                
                # 显示图表
                st.pyplot(fig)
                
                # 显示统计信息
                st.write('情感得分统计：')
                st.write(f'- 均值: {mean_score:.4f}')
                st.write(f'- 中位数: {median_score:.4f}')
                st.write(f'- 标准差: {std_score:.4f}')
                st.write(f'- 最小值: {comments_df["ensemble_sentiment_score"].min():.4f}')
                st.write(f'- 最大值: {comments_df["ensemble_sentiment_score"].max():.4f}')
            else:
                st.write('数据中没有融合情感得分列')
        except Exception as e:
            st.error(f'绘制情感得分分布图时发生错误：{str(e)}')
    
    # 显示平均情感与次日收益率的关系
    st.subheader('情感与收益率关系分析')
    
    try:
        if merged_df.empty:
            st.warning('没有可用的数据进行分析。')
        else:
            # 检查数据是否足够进行分析
            if len(merged_df) < 1:
                st.warning('数据严重不足，仅显示基本数据概览。')
                
                # 显示基本数据信息
                st.write(f'数据日期范围：{merged_df["trade_date"].min().strftime("%Y-%m-%d")} 至 {merged_df["trade_date"].max().strftime("%Y-%m-%d")}')
                st.write(f'有效交易日数量：{len(merged_df)} 个')
                st.write(f'平均情感得分：{merged_df["ensemble_mean"].mean():.4f}')
                st.write(f'平均次日收益率：{merged_df["next_day_return"].mean():.4f}%')
            else:
                # 即使数据有限，也尝试显示基本散点图
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if lag_days > 0:
                    scatter_x = merged_df['ensemble_mean_lag'] if 'ensemble_mean_lag' in merged_df.columns else merged_df['ensemble_mean']
                else:
                    scatter_x = merged_df['ensemble_mean']
                scatter_y = merged_df['next_day_return']
                
                # 过滤掉NaN值
                valid_mask = scatter_x.notna() & scatter_y.notna()
                filtered_x = scatter_x[valid_mask]
                filtered_y = scatter_y[valid_mask]
                
                if len(filtered_x) < 1:
                    st.warning(f'有效样本不足（{len(filtered_x)}个样本），仅显示基本图表。')
                    ax.text(0.5, 0.5, f'仅找到{len(filtered_x)}个有效样本点', transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title('数据不足', fontsize=14)
                else:
                    # 绘制散点图
                    ax.scatter(filtered_x, filtered_y, alpha=0.7, color='blue', s=60)
                    
                    # 添加趋势线
                    if len(filtered_x) >= 2:
                        z = np.polyfit(filtered_x, filtered_y, 1)
                        p = np.poly1d(z)
                        ax.plot(filtered_x, p(filtered_x), "r--", alpha=0.8)
                        
                        # 计算相关系数
                        correlation = np.corrcoef(filtered_x, filtered_y)[0, 1]
                        ax.text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=ax.transAxes, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # 设置图表标题和标签
                    lag_text = f"（{lag_days}天滞后）" if lag_days > 0 else ""
                    ax.set_title(f'平均情感得分与次日收益率关系{lag_text}', fontsize=14)
                    ax.set_xlabel('平均情感得分', fontsize=12)
                    ax.set_ylabel('次日收益率(%)', fontsize=12)
                    
                    # 添加网格线
                    ax.grid(True, alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 显示图表
                st.pyplot(fig)
                
                # 显示统计信息
                if len(filtered_x) >= 1:
                    st.write(f'📊 关系分析统计：')
                    st.write(f'- 有效样本数：{len(filtered_x)} 个')
                    if len(filtered_x) >= 2:
                        st.write(f'- 相关系数：{correlation:.4f}')
                    st.write(f'- 平均情感得分：{filtered_x.mean():.4f}')
                    st.write(f'- 平均次日收益率：{filtered_y.mean():.4f}%')
    except Exception as e:
        st.error(f'绘制情感与收益率关系图时发生错误：{str(e)}')
        st.write('请检查数据格式或尝试调整参数。')
    
    # 回归分析
    st.subheader('回归分析')
    
    try:
        if merged_df.empty or len(merged_df) < 2:
            st.warning('数据不足，无法进行回归分析。')
        else:
            # 准备回归数据
            if lag_days > 0:
                x_col = 'ensemble_mean_lag' if 'ensemble_mean_lag' in merged_df.columns else 'ensemble_mean'
                count_col = 'comment_count_lag' if 'comment_count_lag' in merged_df.columns else 'comment_count'
            else:
                x_col = 'ensemble_mean'
                count_col = 'comment_count'
            
            # 过滤NaN值
            regression_df = merged_df[[x_col, count_col, 'next_day_return']].dropna()
            
            if len(regression_df) < 2:
                st.warning('有效数据不足，无法进行回归分析。')
            else:
                # 单变量回归（情感得分）
                X = regression_df[[x_col]]
                y = regression_df['next_day_return']
                
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                
                # 双变量回归（情感得分 + 评论数量）
                X2 = regression_df[[x_col, count_col]]
                model2 = LinearRegression()
                model2.fit(X2, y)
                r2_2 = model2.score(X2, y)
                
                # 显示回归结果
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('### 单变量回归（情感得分）')
                    st.write(f'- R²: {r2:.4f}')
                    st.write(f'- 回归系数: {model.coef_[0]:.6f}')
                    st.write(f'- 截距: {model.intercept_:.6f}')
                    
                    # 创建回归图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(regression_df[x_col], regression_df['next_day_return'], alpha=0.7)
                    ax.plot(regression_df[x_col], model.predict(X), color='red', linewidth=2)
                    ax.set_xlabel('平均情感得分')
                    ax.set_ylabel('次日收益率(%)')
                    ax.set_title('单变量回归结果')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.write('### 双变量回归（情感得分 + 评论数量）')
                    st.write(f'- R²: {r2_2:.4f}')
                    st.write(f'- 情感得分系数: {model2.coef_[0]:.6f}')
                    st.write(f'- 评论数量系数: {model2.coef_[1]:.6f}')
                    st.write(f'- 截距: {model2.intercept_:.6f}')
                    
                    # 创建系数比较图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    coefficients = [model2.coef_[0], model2.coef_[1]]
                    labels = ['情感得分', '评论数量']
                    colors = ['blue', 'green']
                    bars = ax.bar(labels, coefficients, color=colors)
                    ax.set_ylabel('回归系数')
                    ax.set_title('双变量回归系数比较')
                    ax.grid(True, alpha=0.3)
                    
                    # 添加数值标签
                    for bar, coeff in zip(bars, coefficients):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + (0.000001 if height > 0 else -0.000001),
                               f'{coeff:.6f}', ha='center', va='bottom' if height > 0 else 'top')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 模型比较
                st.write('### 模型比较')
                st.write(f'- 单变量模型 R²: {r2:.4f}')
                st.write(f'- 双变量模型 R²: {r2_2:.4f}')
                improvement = ((r2_2 - r2) / r2 * 100) if r2 != 0 else 0
                st.write(f'- 模型改进: {improvement:.2f}%')
                
                # 结论
                st.write('### 结论')
                if r2 > 0.1:
                    st.write('✅ 情感得分对次日收益率有较强的预测能力')
                elif r2 > 0.05:
                    st.write('⚠️ 情感得分对次日收益率有一定的预测能力')
                else:
                    st.write('❌ 情感得分对次日收益率的预测能力较弱')
                
                if r2_2 > r2:
                    st.write('✅ 加入评论数量后，模型预测能力有所提升')
                else:
                    st.write('⚠️ 加入评论数量后，模型预测能力提升不明显')
    except Exception as e:
        st.error(f'进行回归分析时发生错误：{str(e)}')
        st.write('请检查数据格式或尝试调整参数。')
    
    # 典型评论展示
    st.subheader('典型评论展示')
    
    try:
        if filtered_comments.empty:
            st.warning('没有可用的评论数据。')
        else:
            # 获取积极和消极评论
            positive_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == '积极'].sort_values('llm_sentiment_score', ascending=False)
            negative_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == '消极'].sort_values('llm_sentiment_score', ascending=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🟢 积极评论（情感得分最高）")
                
                # 显示前5条积极评论
                for i, (_, row) in enumerate(positive_comments.head(5).iterrows()):
                    st.write(f"**评论 {i+1}** (得分: {row['llm_sentiment_score']:.3f})")
                    st.write(f"{row['combined_text']}")
                    st.write(f"*发布时间: {row['post_publish_time'].strftime('%Y-%m-%d %H:%M')}*")
                    st.write("---")
            
            with col2:
                st.write("### 🔴 消极评论（情感得分最低）")
                
                # 显示前5条消极评论
                for i, (_, row) in enumerate(negative_comments.head(5).iterrows()):
                    st.write(f"**评论 {i+1}** (得分: {row['llm_sentiment_score']:.3f})")
                    st.write(f"{row['combined_text']}")
                    st.write(f"*发布时间: {row['post_publish_time'].strftime('%Y-%m-%d %H:%M')}*")
                    st.write("---")
            
            # 情感关键词分析
            st.write("### 情感关键词分析")
            
            # 提取积极和消极评论中的关键词
            positive_text = " ".join(positive_comments['combined_text'].tolist())
            negative_text = " ".join(negative_comments['combined_text'].tolist())
            
            # 简单的关键词提取（基于词频）
            # 中文分词简单处理（按字符分割）
            def extract_keywords(text, top_n=10):
                # 移除标点符号和数字
                text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
                # 分割成单词
                words = text.split()
                # 过滤掉单字符和常见停用词
                stop_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
                words = [word for word in words if len(word) > 1 and word not in stop_words]
                # 统计词频
                word_count = Counter(words)
                return word_count.most_common(top_n)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**积极评论关键词**")
                pos_keywords = extract_keywords(positive_text)
                for word, count in pos_keywords:
                    st.write(f"- {word}: {count}次")
            
            with col2:
                st.write("**消极评论关键词**")
                neg_keywords = extract_keywords(negative_text)
                for word, count in neg_keywords:
                    st.write(f"- {word}: {count}次")
            
            # 情感随时间变化
            st.write("### 情感随时间变化")
            
            # 按日期计算平均情感得分
            daily_sentiment_trend = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date)['llm_sentiment_score'].mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_sentiment_trend.plot(ax=ax, marker='o', linestyle='-', linewidth=2, markersize=5)
            
            # 添加零线
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # 设置图表标题和标签
            ax.set_title('每日平均情感得分变化趋势', fontsize=14)
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('平均情感得分', fontsize=12)
            
            # 添加网格线
            ax.grid(True, alpha=0.3)
            
            # 调整日期标签
            plt.xticks(rotation=45, fontsize=10)
            
            # 调整布局
            plt.tight_layout()
            
            # 显示图表
            st.pyplot(fig)
            
            # 显示统计信息
            avg_sentiment = daily_sentiment_trend.mean()
            max_sentiment_date = daily_sentiment_trend.idxmax()
            min_sentiment_date = daily_sentiment_trend.idxmin()
            
            st.write(f"📊 情感趋势统计：")
            st.write(f"- 平均情感得分：{avg_sentiment:.4f}")
            st.write(f"- 最积极日期：{max_sentiment_date}（得分：{daily_sentiment_trend[max_sentiment_date]:.4f}）")
            st.write(f"- 最消极日期：{min_sentiment_date}（得分：{daily_sentiment_trend[min_sentiment_date]:.4f}）")
            
    except Exception as e:
        st.error(f'展示典型评论时发生错误：{str(e)}')
        st.write('请检查数据格式或尝试调整参数。')

except Exception as e:
    st.error(f'应用程序运行时发生错误：{str(e)}')
    st.write('请检查数据文件是否完整，或尝试调整参数。')

# 页脚
st.markdown("---")
st.markdown("### 关于")
st.markdown("本应用基于东方财富股吧评论数据，使用情感分析技术分析投资者情绪与股票收益率之间的关系。")
st.markdown("数据来源：东方财富股吧、股票价格数据")
st.markdown("技术栈：Python、Streamlit、Pandas、Scikit-learn")
