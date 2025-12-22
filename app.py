import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor
import warnings
import os
from matplotlib.font_manager import FontProperties  # 导入字体管理
import matplotlib.font_manager as fm

# 定义全局字体对象
font_prop = None

def setup_chinese_font():
    global font_prop  # 声明使用全局变量
    font_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SourceHanSansSC-Regular.otf")
    
    if os.path.exists(font_file):
        font_prop = FontProperties(fname=font_file)
        # 全局设置字体
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.titlesize"] = 14  # 标题大小
        plt.rcParams["axes.labelsize"] = 12  # 标签大小
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        sns.set(font=font_prop.get_name())  # Seaborn字体设置
    else:
        st.error(f"未找到字体文件：{font_file}")
        # 备用字体设置
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        font_prop = FontProperties(family='WenQuanYi Micro Hei')

# 调用字体设置函数
setup_chinese_font()

warnings.filterwarnings('ignore')

# 加载情感词典
@st.cache_data
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

# 加载数据
@st.cache_data
def load_data(stock_code):
    # 加载评论和情感分析数据
    # 优先使用统一情感分析结果文件
    unified_file = f"{stock_code}_sentiment_analysis_unified.csv"
    updated_file = f"{stock_code}_sentiment_analysis_updated.csv"
    original_file = f"{stock_code}_sentiment_analysis.csv"
    
    if os.path.exists(unified_file):
        comments_df = pd.read_csv(unified_file)
        st.success(f"已加载统一情感分析结果（{len(comments_df)}条评论）")
    elif os.path.exists(updated_file):
        comments_df = pd.read_csv(updated_file)
        st.info(f"已加载改进的情感分析结果（{len(comments_df)}条评论）")
    else:
        comments_df = pd.read_csv(original_file)
        st.warning(f"已加载原始情感分析结果（{len(comments_df)}条评论）")
    
    comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
    
    # 加载价格数据
    price_df = pd.read_csv(f"{stock_code}_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    
    return comments_df, price_df

# 处理数据
def process_data(comments_df, price_df, text_length_limit=500, window_size=30, lag_days=0):
    # 处理combined_text字段为空的情况
    filtered_comments = comments_df.copy()
    
    # 调整文本字段优先级：优先使用post_title（977条非空），再使用combined_text和processed_content
    filtered_comments['combined_text'] = filtered_comments['post_title']
    
    # 过滤无效评论内容
    invalid_pattern = r'(图片图片|转发转发|^[!！]{5,}$|^[?？]{5,}$|^\.{5,}$|^\s*$)'
    filtered_comments = filtered_comments[~filtered_comments['combined_text'].str.contains(invalid_pattern, na=False, regex=True)]
    
    # 检查是否已经包含统一情感分析结果
    if 'lexicon_sentiment' in filtered_comments.columns and 'llm_sentiment_score' in filtered_comments.columns:
        # 已有统一情感分析结果，直接使用
        # 确保所有必要的列都存在
        if 'ensemble_sentiment_score' not in filtered_comments.columns:
            # 如果没有集成法结果，使用LLM法结果
            filtered_comments['ensemble_sentiment_score'] = filtered_comments['llm_sentiment_score']
        
        # 确保情感标签列存在
        if 'llm_sentiment_label' not in filtered_comments.columns:
            # 如果没有情感标签，根据得分生成
            def score_to_label(score):
                if score > 0.1:
                    return '积极'
                elif score < -0.1:
                    return '消极'
                else:
                    return '中性'
            
            filtered_comments['llm_sentiment_label'] = filtered_comments['llm_sentiment_score'].apply(score_to_label)
    else:
        # 需要计算情感分析结果（向后兼容）
        # 加载情感词典
        positive_words, negative_words = load_sentiment_dictionaries()
        
        # 应用基于词典的情感分析
        sentiment_results = filtered_comments['combined_text'].apply(
            lambda x: lexicon_based_sentiment_analysis(x, positive_words, negative_words)
        )
        
        # 将结果拆分为情感标签和得分列
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
    
    # 合并价格数据
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    
    # 处理没有评论的日期（填充NaN值）
    merged_df['comment_count'] = merged_df['comment_count'].fillna(0)
    merged_df['ensemble_mean'] = merged_df['ensemble_mean'].fillna(0)
    merged_df['ensemble_median'] = merged_df['ensemble_median'].fillna(0)
    merged_df['ensemble_std'] = merged_df['ensemble_std'].fillna(0)
    merged_df['llm_mean'] = merged_df['llm_mean'].fillna(0)
    merged_df['lexicon_mean'] = merged_df['lexicon_mean'].fillna(0)
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

# 侧边栏：股票选择（固定为东方财富）
st.sidebar.subheader('股票选择')
stock_code = st.sidebar.selectbox('选择股票代码', ['300059'], index=0)
stock_name = '东方财富'

# 侧边栏：参数调整
st.sidebar.subheader('参数调整')

# 使用session_state管理参数状态
if 'text_length' not in st.session_state:
    st.session_state.text_length = 500
if 'window_size' not in st.session_state:
    st.session_state.window_size = 30
if 'lag_days' not in st.session_state:
    st.session_state.lag_days = 0
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

# 重置按钮
if st.sidebar.button('🔄 重置所有参数'):
    st.session_state.text_length = 500
    st.session_state.window_size = 30
    st.session_state.lag_days = 0
    st.session_state.temperature = 0.1

temperature = st.sidebar.slider('LLM温度参数', 0.0, 1.0, st.session_state.temperature, step=0.1, key='temp_slider')
text_length = st.sidebar.slider('文本长度限制', 50, 1000, st.session_state.text_length, step=50, key='length_slider')
window_size = st.sidebar.slider('移动平均窗口大小(天)', 1, 90, st.session_state.window_size, step=5, key='window_slider')
lag_days = st.sidebar.slider('情感滞后天数', 0, 10, st.session_state.lag_days, step=1, key='lag_slider')

# 更新session_state
st.session_state.text_length = text_length
st.session_state.window_size = window_size
st.session_state.lag_days = lag_days
st.session_state.temperature = temperature

# 加载和处理数据
try:
    comments_df, price_df = load_data(stock_code)
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
                ax.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9, fontproperties=font_prop)
            
            # 设置图表标题和标签（显式指定字体）
            ax.set_title('每日评论数量变化趋势', fontsize=14, fontproperties=font_prop)
            ax.set_xlabel('日期', fontsize=12, fontproperties=font_prop)
            ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
            
            # 调整Y轴范围
            ax.set_ylim(0, daily_comments.max() * 1.1)
            
            # 添加网格线
            ax.grid(True, alpha=0.3)
            
            # 调整日期标签
            plt.xticks(rotation=45, fontsize=10, fontproperties=font_prop)
            plt.yticks(fontproperties=font_prop)
            
            # 计算统计信息
            avg_daily = daily_comments.mean()
            max_daily = daily_comments.max()
            min_daily = daily_comments.min()
            
            # 在图表中添加统计信息
            stats_text = f'平均日评论数: {avg_daily:.1f}\n最高日评论数: {max_daily}\n最低日评论数: {min_daily}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
                    fontsize=10, fontproperties=font_prop)
        else:
            ax.set_title('暂无评论数据', fontsize=14, fontproperties=font_prop)
            ax.text(0.5, 0.5, '没有足够的评论数据来绘制趋势图', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, fontproperties=font_prop)
        
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
    
    # 添加情感分析统计表格
    if 'lexicon_sentiment' in comments_df.columns and 'llm_sentiment_score' in comments_df.columns and 'ensemble_sentiment_score' in comments_df.columns:
        st.write('### 情感分析方法比较')
        
        # 计算三种方法的统计指标
        methods_stats = pd.DataFrame({
            '方法': ['词典法', 'LLM法', '集成法'],
            '平均情感得分': [
                comments_df['lexicon_sentiment'].mean(),
                comments_df['llm_sentiment_score'].mean(),
                comments_df['ensemble_sentiment_score'].mean()
            ],
            '标准差': [
                comments_df['lexicon_sentiment'].std(),
                comments_df['llm_sentiment_score'].std(),
                comments_df['ensemble_sentiment_score'].std()
            ]
        })
        
        # 计算积极、中性、消极比例
        if 'llm_sentiment_label' in comments_df.columns:
            sentiment_counts = comments_df['llm_sentiment_label'].value_counts()
            total_comments = len(comments_df)
            
            # 词典法比例（基于得分计算）
            lexicon_positive = (comments_df['lexicon_sentiment'] > 0.1).sum()
            lexicon_negative = (comments_df['lexicon_sentiment'] < -0.1).sum()
            lexicon_neutral = total_comments - lexicon_positive - lexicon_negative
            
            # LLM法比例（基于标签计算）
            llm_positive = sentiment_counts.get('积极', 0)
            llm_negative = sentiment_counts.get('消极', 0)
            llm_neutral = sentiment_counts.get('中性', 0)
            
            # 集成法比例（基于得分计算）
            ensemble_positive = (comments_df['ensemble_sentiment_score'] > 0.1).sum()
            ensemble_negative = (comments_df['ensemble_sentiment_score'] < -0.1).sum()
            ensemble_neutral = total_comments - ensemble_positive - ensemble_negative
            
            # 添加比例列
            methods_stats['积极比例'] = [
                f"{lexicon_positive/total_comments*100:.2f}%",
                f"{llm_positive/total_comments*100:.2f}%",
                f"{ensemble_positive/total_comments*100:.2f}%"
            ]
            methods_stats['中性比例'] = [
                f"{lexicon_neutral/total_comments*100:.2f}%",
                f"{llm_neutral/total_comments*100:.2f}%",
                f"{ensemble_neutral/total_comments*100:.2f}%"
            ]
            methods_stats['消极比例'] = [
                f"{lexicon_negative/total_comments*100:.2f}%",
                f"{llm_negative/total_comments*100:.2f}%",
                f"{ensemble_negative/total_comments*100:.2f}%"
            ]
        
        # 显示统计表格
        st.table(methods_stats.set_index('方法'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('### 情感标签分布')
        try:
            if 'llm_sentiment_label' in comments_df.columns:
                sentiment_counts = comments_df['llm_sentiment_label'].value_counts()
                
                if len(sentiment_counts) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#4caf50' if label == '积极' else '#ff9800' if label == '中性' else '#f44336' for label in sentiment_counts.index]
                    explode = [0.1 if label in ['消极', '积极'] else 0 for label in sentiment_counts.index]
                    
                    # 绘制饼图，仅接收扇形对象
                    patches, _ = ax.pie(
                        sentiment_counts.values, 
                        startangle=90, 
                        colors=colors, 
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}, 
                        explode=explode
                    )
                    
                    # 遍历每个扇形，单独处理“消极”“积极”
                    for i, label in enumerate(sentiment_counts.index):
                        if label == '消极':
                            patch = patches[i]
                            # 计算消极扇形的中间角度
                            theta_mid = (patch.theta1 + patch.theta2) / 2
                            # 计算指向消极区域中心的坐标（半径缩小，更靠近图表）
                            r = patch.r * 0.9
                            x = r * np.cos(np.radians(theta_mid))
                            y = r * np.sin(np.radians(theta_mid))
                            # 标签位置靠近图表（xytext调整为更紧凑）
                            ax.annotate(
                                f'消极 ({sentiment_counts[label]/len(comments_df)*100:.1f}%)',
                                xy=(x, y),  # 箭头指向消极区域中心
                                xytext=(1.2, 0.7),  # 标签靠近图表
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                                fontsize=11,
                                fontproperties=font_prop
                            )
                        elif label == '积极':
                            patch = patches[i]
                            # 计算积极扇形的中间角度
                            theta_mid = (patch.theta1 + patch.theta2) / 2
                            # 计算指向积极区域中心的坐标
                            r = patch.r * 0.9
                            x = r * np.cos(np.radians(theta_mid))
                            y = r * np.sin(np.radians(theta_mid))
                            # 标签位置与消极错开，靠近图表
                            ax.annotate(
                                f'积极 ({sentiment_counts[label]/len(comments_df)*100:.1f}%)',
                                xy=(x, y),  # 箭头指向积极区域中心
                                xytext=(1.2, 0.5),  # 标签靠近图表
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                                fontsize=11,
                                fontproperties=font_prop
                            )
                    
                    # 中性标签
                    ax.text(0, -1.2, f'中性 ({sentiment_counts["中性"]/len(comments_df)*100:.1f}%)', ha='center', fontsize=12, fontproperties=font_prop)
                    
                    ax.set_title('LLM情感标签分布', fontsize=14, fontproperties=font_prop)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    # 显示数量
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
                
                # 设置图表标题和标签（显式指定字体）
                ax.set_title('融合情感得分分布', fontsize=14, fontproperties=font_prop)
                ax.set_xlabel('情感得分', fontsize=12, fontproperties=font_prop)
                ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
                
                # 添加网格线
                ax.grid(True, alpha=0.3)
                
                # 添加图例（显式指定字体）
                ax.legend(prop=font_prop)
                
                # 调整刻度字体
                plt.xticks(fontproperties=font_prop)
                plt.yticks(fontproperties=font_prop)
                
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
            if len(merged_df) < 1:
                st.warning('数据严重不足，仅显示基本数据概览。')
                st.write(f'数据日期范围：{merged_df["trade_date"].min().strftime("%Y-%m-%d")} 至 {merged_df["trade_date"].max().strftime("%Y-%m-%d")}')
                st.write(f'有效交易日数量：{len(merged_df)} 个')
                st.write(f'平均情感得分：{merged_df["ensemble_mean"].mean():.4f}')
                st.write(f'平均次日收益率：{merged_df["next_day_return"].mean():.4f}%')
            else:
                # 创建散点图
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
                    ax.text(0.5, 0.5, f'仅找到{len(filtered_x)}个有效样本点', transform=ax.transAxes, 
                            ha='center', va='center', fontsize=12, fontproperties=font_prop)
                    ax.set_title('数据不足', fontsize=14, fontproperties=font_prop)
                else:
                    # 根据情感得分设置不同颜色
                    colors = ['red' if s < -0.1 else 'green' if s > 0.1 else 'blue' for s in filtered_x]
                    ax.scatter(filtered_x, filtered_y, c=colors, alpha=0.5)
                    ax.set_title(f'平均情感得分与次日收益率关系 (滞后{lag_days}天)', fontsize=14, fontproperties=font_prop)
                    ax.set_xlabel(f'平均情感得分(滞后{lag_days}天)' if lag_days > 0 else '平均情感得分', fontsize=12, fontproperties=font_prop)
                    ax.set_ylabel('次日收益率 (%)', fontsize=12, fontproperties=font_prop)
                    ax.grid(True, alpha=0.3)
                    
                    # 调整刻度字体
                    plt.xticks(fontproperties=font_prop)
                    plt.yticks(fontproperties=font_prop)
                    
                    # 尝试简单的线性回归
                    try:
                        if len(filtered_x) >= 2:
                            X_simple = filtered_x.values.reshape(-1, 1)
                            y_simple = filtered_y.values
                            model = LinearRegression()
                            model.fit(X_simple, y_simple)
                            r2_score = model.score(X_simple, y_simple)
                            
                            # 绘制回归线
                            x_line = np.linspace(filtered_x.min(), filtered_x.max(), 100).reshape(-1, 1)
                            y_line = model.predict(x_line)
                            ax.plot(x_line, y_line, color='red', label=f'简单回归线 (R²={r2_score:.3f})')
                            ax.legend(prop=font_prop)
                    except Exception as e:
                        pass
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 添加图表说明
                st.write('📊 图表说明：')
                st.write('- 绿色点：积极情感得分 (> 0.1)')
                st.write('- 蓝色点：中性情感得分 (± 0.1)')
                st.write('- 红色点：消极情感得分 (< -0.1)')
                st.write('- 红色线：简单回归线 (如适用)')
                
                # 显示基本统计信息
                st.subheader('基本统计信息')
                st.write(f'总交易日数量：{len(merged_df)} 个')
                st.write(f'有评论的交易日数量：{sum(merged_df["comment_count"] > 0)} 个')
                st.write(f'平均每日评论数：{merged_df["comment_count"].mean():.2f} 条')
                st.write(f'平均情感得分：{merged_df["ensemble_mean"].mean():.4f}')
                st.write(f'平均次日收益率：{merged_df["next_day_return"].mean():.4f}%')
                
                # 详细回归分析
                if len(merged_df) >= 3:
                    try:
                        if lag_days > 0:
                            required_cols = ['ensemble_mean_lag', 'comment_count_lag', 'ensemble_std_lag']
                            if not all(col in merged_df.columns for col in required_cols):
                                st.info(f'滞后{lag_days}天的数据不足，使用非滞后数据进行分析。')
                                X = merged_df[['ensemble_mean', 'comment_count', 'ensemble_std']]
                                current_lag = 0
                            else:
                                X = merged_df[required_cols]
                                current_lag = lag_days
                        else:
                            X = merged_df[['ensemble_mean', 'comment_count', 'ensemble_std']]
                            current_lag = 0
                        y = merged_df['next_day_return']
                        
                        valid_mask = X.notna().all(axis=1) & y.notna()
                        X_valid = X[valid_mask]
                        y_valid = y[valid_mask]
                        
                        if len(X_valid) >= 3:
                            st.subheader('回归分析结果')
                            
                            try:
                                model = LinearRegression()
                                model.fit(X_valid, y_valid)
                                r2_score = model.score(X_valid, y_valid)
                                
                                st.write('**标准线性回归**')
                                st.write(f'R²值: {r2_score:.4f}')
                                st.write(f'截距: {model.intercept_:.4f}')
                                
                                if current_lag > 0:
                                    st.write(f'滞后情感系数: {model.coef_[0]:.4f}')
                                    st.write(f'滞后评论数系数: {model.coef_[1]:.4f}')
                                    st.write(f'滞后情感波动系数: {model.coef_[2]:.4f}')
                                else:
                                    st.write(f'情感系数: {model.coef_[0]:.4f}')
                                    st.write(f'评论数系数: {model.coef_[1]:.4f}')
                                    st.write(f'情感波动系数: {model.coef_[2]:.4f}')
                            except Exception as e:
                                st.info(f'多变量回归失败: {str(e)}，尝试单变量回归。')
                                
                                X_simple = X_valid[[X_valid.columns[0]]]
                                model = LinearRegression()
                                model.fit(X_simple, y_valid)
                                r2_score = model.score(X_simple, y_valid)
                                
                                st.write('**单变量线性回归**')
                                st.write(f'R²值: {r2_score:.4f}')
                                st.write(f'截距: {model.intercept_:.4f}')
                                st.write(f'{X_simple.columns[0]}系数: {model.coef_[0]:.4f}')
                            
                            st.info(f'💡 回归分析解释：')
                            st.write(f'- R²值越接近1，表示模型拟合效果越好')
                            st.write(f'- 情感系数为正，表示情感越积极，次日收益率可能越高')
                    except Exception as e:
                        st.info(f'详细回归分析不可用: {str(e)}')
    except Exception as e:
        st.error(f'进行情感与收益率关系分析时发生错误：{str(e)}')
        if not merged_df.empty:
            st.write('📊 基本数据概览：')
            st.write(f'数据日期范围：{merged_df["trade_date"].min().strftime("%Y-%m-%d")} 至 {merged_df["trade_date"].max().strftime("%Y-%m-%d")}')
            st.write(f'有效交易日数量：{len(merged_df)} 个')
            st.write(f'平均情感得分：{merged_df["ensemble_mean"].mean():.4f}')
            st.write(f'平均次日收益率：{merged_df["next_day_return"].mean():.4f}%')
        else:
            st.write('无法获取有效数据进行分析。')
    
    # 显示评论示例
    st.subheader('评论示例')
    selected_sentiment = st.selectbox('选择情感类型', ['积极', '中性', '消极'])
    sentiment_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == selected_sentiment]
    if len(sentiment_comments) > 0:
        st.dataframe(sentiment_comments[['post_publish_time', 'combined_text']].sample(min(10, len(sentiment_comments))))
    else:
        st.write(f'没有找到{selected_sentiment}情感类型的评论示例。')
    
    # 参数影响分析
    st.subheader('当前参数影响分析')
        
    st.write(f'📝 文本长度限制: {text_length} 字符（过滤掉 {len(comments_df) - len(filtered_comments)} 条长评论）')
    st.write(f'📊 移动平均窗口: {window_size} 天（平滑情感和收益率数据）')
    st.write(f'⏱️ 情感滞后天数: {lag_days} 天（分析情感对未来 {lag_days} 天收益率的影响）')
    st.write(f'🎲 LLM温度参数: {temperature}（影响模型生成的随机性，值越高生成内容越多样）')
    
    st.info('💡 提示：调整任何参数后，应用将自动重新运行并更新所有分析结果。')

except Exception as e:
    st.error(f'发生错误: {e}')
    st.write('请检查数据文件是否存在或格式是否正确。')
