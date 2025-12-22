import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
import os
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import re

# 全局设置
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False  # 提前解决负号显示问题

# 定义全局字体对象
font_prop = None

def setup_chinese_font():
    """优化的中文字体设置，增加更多备用字体"""
    global font_prop
    # 优先尝试的字体列表
    font_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "SourceHanSansSC-Regular.otf"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimHei.ttf"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Microsoft YaHei.ttf")
    ]
    
    # 系统内置备用字体
    system_fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
    
    # 尝试加载本地字体文件
    for font_file in font_paths:
        if os.path.exists(font_file):
            try:
                font_prop = FontProperties(fname=font_file)
                # 全局设置字体
                plt.rcParams["font.family"] = font_prop.get_name()
                sns.set(font=font_prop.get_name())
                st.success(f"成功加载本地字体：{font_file}")
                return
            except Exception as e:
                st.warning(f"加载字体文件失败：{e}")
                continue
    
    # 尝试使用系统字体
    for font_name in system_fonts:
        try:
            font_prop = FontProperties(family=font_name)
            plt.rcParams['font.sans-serif'] = [font_name]
            sns.set(font=font_name)
            st.info(f"使用系统备用字体：{font_name}")
            return
        except Exception:
            continue
    
    # 最终兜底
    st.warning("未找到合适的中文字体，可能导致中文显示异常")
    font_prop = FontProperties(family='DejaVu Sans')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 调用字体设置
setup_chinese_font()

# 加载情感词典
@st.cache_data
def load_sentiment_dictionaries():
    """加载情感词典，增加异常处理"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_dict_path = os.path.join(script_dir, 'zhang_unformal_pos (1).txt')
    neg_dict_path = os.path.join(script_dir, 'zhang_unformal_neg (1).txt')
    
    # 初始化空列表
    positive_words = []
    negative_words = []
    
    # 加载积极词典
    try:
        with open(pos_dict_path, 'r', encoding='utf-8') as f:
            positive_words = [line.strip() for line in f if line.strip()]
        st.success(f"成功加载积极词典，共{len(positive_words)}个词汇")
    except FileNotFoundError:
        st.error(f"未找到积极词典文件：{pos_dict_path}")
    except Exception as e:
        st.error(f"加载积极词典失败：{e}")
    
    # 加载消极词典
    try:
        with open(neg_dict_path, 'r', encoding='utf-8') as f:
            negative_words = [line.strip() for line in f if line.strip()]
        st.success(f"成功加载消极词典，共{len(negative_words)}个词汇")
    except FileNotFoundError:
        st.error(f"未找到消极词典文件：{neg_dict_path}")
    except Exception as e:
        st.error(f"加载消极词典失败：{e}")
    
    return positive_words, negative_words

# 优化的情感分析函数
def lexicon_based_sentiment_analysis(text, pos_words, neg_words):
    """
    优化的基于词典的情感分析
    降低判定阈值，减少过度中性化
    """
    if pd.isna(text) or text.strip() == '':
        return '中性', 0.0
    
    # 转换为字符串并清理
    text = str(text).strip()
    
    # 计算情感词出现次数
    pos_count = sum(1 for word in pos_words if word in text)
    neg_count = sum(1 for word in neg_words if word in text)
    
    # 优化得分计算：避免+1稀释，用max防止除0
    total = max(pos_count + neg_count, 1)
    sentiment_score = (pos_count - neg_count) / total
    
    # 降低判定阈值，减少中性比例
    if sentiment_score > 0.05:
        sentiment_label = '积极'
    elif sentiment_score < -0.05:
        sentiment_label = '消极'
    else:
        sentiment_label = '中性'
    
    return sentiment_label, sentiment_score

# 设置页面配置
st.set_page_config(
    page_title="东方财富股吧评论情感分析",
    page_icon="📈",
    layout="wide"
)

# 页面标题
st.title('📈 东方财富股吧评论情感分析')

# 加载数据
@st.cache_data
def load_data(stock_code):
    """优化的数据加载逻辑，增加文件检查"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 定义文件路径
    file_map = {
        "unified": os.path.join(script_dir, f"{stock_code}_sentiment_analysis_unified.csv"),
        "updated": os.path.join(script_dir, f"{stock_code}_sentiment_analysis_updated.csv"),
        "original": os.path.join(script_dir, f"{stock_code}_sentiment_analysis.csv")
    }
    
    # 检查文件并加载
    for file_type, file_path in file_map.items():
        if os.path.exists(file_path):
            try:
                comments_df = pd.read_csv(file_path)
                st.success(f"已加载{file_type}情感分析结果（{len(comments_df)}条评论）")
                break
            except Exception as e:
                st.error(f"加载{file_type}文件失败：{e}")
                continue
    else:
        st.error("未找到任何情感分析数据文件")
        return pd.DataFrame(), pd.DataFrame()
    
    # 加载价格数据
    price_path = os.path.join(script_dir, f"{stock_code}_price_data.csv")
    try:
        price_df = pd.read_csv(price_path)
    except FileNotFoundError:
        st.error(f"未找到价格数据文件：{price_path}")
        price_df = pd.DataFrame()
    except Exception as e:
        st.error(f"加载价格数据失败：{e}")
        price_df = pd.DataFrame()
    
    # 处理日期列
    if not comments_df.empty:
        comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'], errors='coerce')
    if not price_df.empty:
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'], errors='coerce')
    
    return comments_df, price_df

# 数据处理函数
def process_data(comments_df, price_df, text_length_limit=500, window_size=30, lag_days=0):
    """优化的数据处理逻辑"""
    if comments_df.empty or price_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    filtered_comments = comments_df.copy()
    
    # 调整文本字段优先级
    text_fields = ['post_title', 'combined_text', 'processed_content']
    for field in text_fields:
        if field in filtered_comments.columns:
            filtered_comments['combined_text'] = filtered_comments[field]
            break
    
    # 过滤无效评论（优化正则）
    invalid_pattern = r'(图片图片|转发转发|^[!！]{3,}$|^[?？]{3,}$|^\\.{3,}$|^\\s*$|^转发$|^图片$)'
    filtered_comments = filtered_comments[~filtered_comments['combined_text'].astype(str).str.contains(invalid_pattern, na=True, regex=True)]
    
    # 情感分析处理
    if 'lexicon_sentiment' not in filtered_comments.columns or 'llm_sentiment_score' not in filtered_comments.columns:
        # 加载词典并计算情感
        positive_words, negative_words = load_sentiment_dictionaries()
        if positive_words and negative_words:
            sentiment_results = filtered_comments['combined_text'].apply(
                lambda x: lexicon_based_sentiment_analysis(x, positive_words, negative_words)
            )
            filtered_comments['llm_sentiment_label'] = sentiment_results.str[0]
            filtered_comments['llm_sentiment_score'] = sentiment_results.str[1]
            filtered_comments['ensemble_sentiment_score'] = sentiment_results.str[1]
            filtered_comments['lexicon_sentiment'] = sentiment_results.str[1]
        else:
            st.error("情感词典加载失败，无法进行情感分析")
            return pd.DataFrame(), pd.DataFrame()
    else:
        # 确保标签列存在（优化阈值）
        if 'llm_sentiment_label' not in filtered_comments.columns:
            def score_to_label(score):
                if score > 0.05:
                    return '积极'
                elif score < -0.05:
                    return '消极'
                else:
                    return '中性'
            filtered_comments['llm_sentiment_label'] = filtered_comments['llm_sentiment_score'].apply(score_to_label)
    
    # 文本长度过滤
    filtered_comments['text_length'] = filtered_comments['combined_text'].astype(str).str.len()
    filtered_comments = filtered_comments[
        (filtered_comments['text_length'] >= 1) & 
        (filtered_comments['text_length'] <= text_length_limit)
    ]
    
    # 按日期聚合情感数据
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg(
        ensemble_mean=('ensemble_sentiment_score', 'mean'),
        ensemble_median=('ensemble_sentiment_score', 'median'),
        ensemble_std=('ensemble_sentiment_score', 'std'),
        comment_count=('ensemble_sentiment_score', 'count'),
        llm_mean=('llm_sentiment_score', 'mean'),
        lexicon_mean=('lexicon_sentiment', 'mean')
    ).reset_index()
    
    daily_sentiment.columns = ['date', 'ensemble_mean', 'ensemble_median', 'ensemble_std', 
                              'comment_count', 'llm_mean', 'lexicon_mean']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 合并价格数据
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    
    # 填充缺失值
    fill_cols = ['comment_count', 'ensemble_mean', 'ensemble_median', 'ensemble_std', 'llm_mean', 'lexicon_mean']
    merged_df[fill_cols] = merged_df[fill_cols].fillna(0)
    
    # 添加滞后数据
    if lag_days > 0:
        merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean'].shift(lag_days).fillna(0)
        merged_df['comment_count_lag'] = merged_df['comment_count'].shift(lag_days).fillna(0)
        merged_df['ensemble_std_lag'] = merged_df['ensemble_std'].shift(lag_days).fillna(0)
    
    # 移动平均
    if window_size > 1:
        merged_df['ensemble_mean_rolling'] = merged_df['ensemble_mean'].rolling(window=window_size, min_periods=1).mean()
        if 'next_day_return' in merged_df.columns:
            merged_df['next_day_return_rolling'] = merged_df['next_day_return'].rolling(window=window_size, min_periods=1).mean()
    
    return merged_df, filtered_comments

# 侧边栏设置
st.sidebar.header('⚙️ 参数设置')

# 股票选择
stock_code = st.sidebar.selectbox('选择股票代码', ['300059'], index=0)
stock_name = '东方财富'

# 参数调整（优化默认值）
st.sidebar.subheader('分析参数')
if 'params' not in st.session_state:
    st.session_state.params = {
        'text_length': 500,
        'window_size': 15,
        'lag_days': 1,
        'temperature': 0.1
    }

# 重置按钮
if st.sidebar.button('🔄 重置参数'):
    st.session_state.params = {
        'text_length': 500,
        'window_size': 15,
        'lag_days': 1,
        'temperature': 0.1
    }

# 滑块设置
temperature = st.sidebar.slider(
    'LLM温度参数', 0.0, 1.0, 
    st.session_state.params['temperature'], 0.1, key='temp'
)
text_length = st.sidebar.slider(
    '文本长度限制', 50, 1000, 
    st.session_state.params['text_length'], 50, key='length'
)
window_size = st.sidebar.slider(
    '移动平均窗口(天)', 1, 90, 
    st.session_state.params['window_size'], 5, key='window'
)
lag_days = st.sidebar.slider(
    '情感滞后天数', 0, 10, 
    st.session_state.params['lag_days'], 1, key='lag'
)

# 更新session state
st.session_state.params.update({
    'text_length': text_length,
    'window_size': window_size,
    'lag_days': lag_days,
    'temperature': temperature
})

# 主分析逻辑
try:
    # 加载数据
    comments_df, price_df = load_data(stock_code)
    
    if comments_df.empty or price_df.empty:
        st.error("数据加载失败，请检查文件是否存在")
    else:
        # 处理数据
        merged_df, filtered_comments = process_data(
            comments_df, price_df, 
            text_length, window_size, lag_days
        )
        
        # 数据质量检查
        st.header('📊 数据质量检查')
        col1, col2, col3, col4 = st.columns(4)
        
        total_comments = len(comments_df)
        filtered_count = len(filtered_comments)
        zero_sentiment = (filtered_comments['ensemble_sentiment_score'] == 0).sum()
        valid_days = len(merged_df[merged_df['comment_count'] > 0])
        
        with col1:
            st.metric("总评论数", total_comments)
        with col2:
            st.metric("有效评论数", filtered_count)
        with col3:
            st.metric("零情感得分评论", zero_sentiment)
        with col4:
            st.metric("有评论的交易日", valid_days)
        
        # 警告提示
        if filtered_count / total_comments < 0.5:
            st.warning(f"⚠️ 超过50%的评论被过滤，可能影响分析结果")
        if zero_sentiment / filtered_count > 0.8:
            st.warning(f"⚠️ 超过80%的有效评论情感得分为0，情感区分度较低")
        
        # 评论数量趋势
        st.header('📝 评论数量分析')
        try:
            daily_comments = comments_df.groupby(comments_df['post_publish_time'].dt.date)['post_id'].count()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            if not daily_comments.empty:
                daily_comments.plot(ax=ax, marker='o', linewidth=2, markersize=4, color='#2E86AB')
                ax.set_title('每日评论数量变化趋势', fontproperties=font_prop, fontsize=14)
                ax.set_xlabel('日期', fontproperties=font_prop)
                ax.set_ylabel('评论数量', fontproperties=font_prop)
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                stats_text = f'平均：{daily_comments.mean():.1f}条\n最高：{daily_comments.max()}条\n最低：{daily_comments.min()}条'
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontproperties=font_prop)
                
                plt.xticks(rotation=45, fontproperties=font_prop)
                plt.yticks(fontproperties=font_prop)
            else:
                ax.text(0.5, 0.5, '暂无评论数据', transform=ax.transAxes, 
                        ha='center', va='center', fontproperties=font_prop, fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘制评论趋势图失败：{e}")
        
        # 情感分析结果
        st.header('❤️ 情感分析结果')
        col1, col2 = st.columns(2)
        
        # 情感标签分布
        with col1:
            st.subheader('情感标签分布')
            try:
                sentiment_counts = filtered_comments['llm_sentiment_label'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = {'积极': '#27AE60', '中性': '#F39C12', '消极': '#E74C3C'}
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values,
                    labels=sentiment_counts.index,
                    colors=[colors.get(l, '#95A5A6') for l in sentiment_counts.index],
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=(0.05, 0.05, 0.05)
                )
                
                # 设置饼图字体
                for text in texts + autotexts:
                    text.set_fontproperties(font_prop)
                ax.set_title('LLM情感标签分布', fontproperties=font_prop, fontsize=14)
                
                st.pyplot(fig)
                # 显示详细数据
                st.write("详细统计：")
                for label, count in sentiment_counts.items():
                    st.write(f"- {label}：{count}条 ({count/len(filtered_comments)*100:.1f}%)")
            except Exception as e:
                st.error(f"绘制情感分布饼图失败：{e}")
        
        # 情感得分分布
        with col2:
            st.subheader('情感得分分布')
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(
                    filtered_comments['ensemble_sentiment_score'],
                    bins=30,
                    kde=True,
                    ax=ax,
                    color='#3498DB',
                    edgecolor='white'
                )
                
                # 添加统计线
                mean_score = filtered_comments['ensemble_sentiment_score'].mean()
                median_score = filtered_comments['ensemble_sentiment_score'].median()
                ax.axvline(mean_score, color='red', linestyle='--', label=f'均值: {mean_score:.3f}')
                ax.axvline(median_score, color='green', linestyle='--', label=f'中位数: {median_score:.3f}')
                
                # 设置标签
                ax.set_title('融合情感得分分布', fontproperties=font_prop, fontsize=14)
                ax.set_xlabel('情感得分', fontproperties=font_prop)
                ax.set_ylabel('评论数量', fontproperties=font_prop)
                ax.legend(prop=font_prop)
                ax.grid(True, alpha=0.3)
                
                plt.xticks(fontproperties=font_prop)
                plt.yticks(fontproperties=font_prop)
                st.pyplot(fig)
                
                # 显示统计信息
                st.write("得分统计：")
                st.write(f"- 均值：{mean_score:.4f}")
                st.write(f"- 中位数：{median_score:.4f}")
                st.write(f"- 标准差：{filtered_comments['ensemble_sentiment_score'].std():.4f}")
            except Exception as e:
                st.error(f"绘制情感得分分布图失败：{e}")
        
        # 情感与收益率关系
        st.header('📈 情感与收益率关系分析')
        try:
            if 'next_day_return' in merged_df.columns and not merged_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 准备数据
                if lag_days > 0 and 'ensemble_mean_lag' in merged_df.columns:
                    x_data = merged_df['ensemble_mean_lag']
                    x_label = f'平均情感得分(滞后{lag_days}天)'
                else:
                    x_data = merged_df['ensemble_mean']
                    x_label = '平均情感得分'
                
                y_data = merged_df['next_day_return']
                valid_mask = x_data.notna() & y_data.notna()
                x_valid = x_data[valid_mask]
                y_valid = y_data[valid_mask]
                
                if len(x_valid) > 0:
                    # 绘制散点图
                    colors = ['#E74C3C' if x < -0.05 else '#27AE60' if x > 0.05 else '#F39C12' for x in x_valid]
                    ax.scatter(x_valid, y_valid, c=colors, alpha=0.6, s=50)
                    
                    # 线性回归
                    if len(x_valid) >= 2:
                        X = x_valid.values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, y_valid)
                        r2 = model.score(X, y_valid)
                        
                        # 绘制回归线
                        x_line = np.linspace(x_valid.min(), x_valid.max(), 100).reshape(-1, 1)
                        y_line = model.predict(x_line)
                        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'回归线 (R²={r2:.3f})')
                    
                    ax.set_title(f'情感得分与次日收益率关系', fontproperties=font_prop, fontsize=14)
                    ax.set_xlabel(x_label, fontproperties=font_prop)
                    ax.set_ylabel('次日收益率 (%)', fontproperties=font_prop)
                    ax.legend(prop=font_prop)
                    ax.grid(True, alpha=0.3)
                    
                    plt.xticks(fontproperties=font_prop)
                    plt.yticks(fontproperties=font_prop)
                else:
                    ax.text(0.5, 0.5, '暂无有效数据', transform=ax.transAxes, 
                            ha='center', va='center', fontproperties=font_prop, fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 回归分析
                st.subheader('📊 回归分析结果')
                if len(x_valid) >= 3:
                    # 多变量回归
                    try:
                        if lag_days > 0:
                            features = ['ensemble_mean_lag', 'comment_count_lag', 'ensemble_std_lag']
                        else:
                            features = ['ensemble_mean', 'comment_count', 'ensemble_std']
                        
                        # 确保特征列存在
                        features = [f for f in features if f in merged_df.columns]
                        X = merged_df[features][valid_mask]
                        y = merged_df['next_day_return'][valid_mask]
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        r2 = model.score(X, y)
                        
                        st.write(f"**多变量线性回归 (R² = {r2:.4f})**")
                        st.write(f"截距：{model.intercept_:.4f}")
                        for i, feat in enumerate(features):
                            st.write(f"{feat} 系数：{model.coef_[i]:.4f}")
                    except Exception as e:
                        st.info(f"多变量回归失败，尝试单变量回归：{e}")
                        
                        # 单变量回归
                        X_simple = x_valid.values.reshape(-1, 1)
                        model_simple = LinearRegression()
                        model_simple.fit(X_simple, y_valid)
                        r2_simple = model_simple.score(X_simple, y_valid)
                        
                        st.write(f"**单变量线性回归 (R² = {r2_simple:.4f})**")
                        st.write(f"截距：{model_simple.intercept_:.4f}")
                        st.write(f"情感得分系数：{model_simple.coef_[0]:.4f}")
            else:
                st.warning("数据中缺少次日收益率列，无法进行相关性分析")
        except Exception as e:
            st.error(f"分析情感与收益率关系失败：{e}")
        
        # 评论示例
        st.header('🔍 评论示例')
        selected_sentiment = st.selectbox('选择情感类型', ['积极', '中性', '消极'])
        sentiment_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == selected_sentiment]
        
        if len(sentiment_comments) > 0:
            sample_comments = sentiment_comments[['post_publish_time', 'combined_text']].sample(min(10, len(sentiment_comments)))
            st.dataframe(sample_comments, use_container_width=True)
        else:
            st.info(f"未找到{selected_sentiment}情感的评论示例")
        
        # 参数影响分析
        st.header('📋 参数影响分析')
        st.write(f"- **文本长度限制**：{text_length}字符，过滤掉{len(comments_df)-len(filtered_comments)}条超长/超短评论")
        st.write(f"- **移动平均窗口**：{window_size}天，用于平滑情感和收益率数据")
        st.write(f"- **情感滞后天数**：{lag_days}天，分析{lag_days}天前的情感对当日收益率的影响")
        st.write(f"- **LLM温度参数**：{temperature}，值越高生成的情感分析结果越多样")
        
except Exception as e:
    st.error(f"程序执行出错：{str(e)}")
    st.info("请检查：1. 数据文件是否存在 2. 词典文件是否正确 3. 依赖库是否安装完整")

# 页脚
st.markdown("---")
st.markdown("© 2025 东方财富股吧评论情感分析工具 | 基于Streamlit开发")
