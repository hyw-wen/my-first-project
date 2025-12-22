import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor
import warnings
import os
from matplotlib.font_manager import FontProperties

# 字体配置
font_prop = None
def setup_chinese_font():
    global font_prop
    font_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SourceHanSansSC-Regular.otf")
    if os.path.exists(font_file):
        font_prop = FontProperties(fname=font_file)
        plt.rcParams.update({
            "font.family": font_prop.get_name(),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.unicode_minus": False
        })
        sns.set(font=font_prop.get_name())
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        font_prop = FontProperties(family='WenQuanYi Micro Hei')

setup_chinese_font()
warnings.filterwarnings('ignore')


# 加载数据（仅过滤空文本）
@st.cache_data
def load_data(stock_code):
    # 加载评论文件
    improved_file = f"{stock_code}_sentiment_analysis_improved_sentiment_analysis.csv"
    if not os.path.exists(improved_file):
        st.error(f"未找到评论文件：{improved_file}")
        st.stop()
    comments_df = pd.read_csv(improved_file)
    comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
    # 仅过滤空文本，不限制长度
    comments_df = comments_df[comments_df['combined_text'].str.strip() != '']
    
    # 加载价格文件
    price_df = pd.read_csv(f"{stock_code}_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    price_df['next_day_return'] = price_df['next_day_return'].fillna(0)
    
    return comments_df, price_df


# 数据处理（取消长度过滤）
def process_data(comments_df, price_df, lag_days=1):
    filtered_comments = comments_df.copy()
    # 复用文件中的情感字段
    filtered_comments['llm_sentiment_label'] = filtered_comments['llm_sentiment_label_new']
    filtered_comments['llm_sentiment_score'] = filtered_comments['llm_sentiment_score_new']
    
    # 按日期聚合
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg({
        'llm_sentiment_score': ['mean', 'count']
    }).reset_index()
    daily_sentiment.columns = ['date', 'llm_mean', 'comment_count']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 合并价格数据
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    merged_df = merged_df.fillna({'llm_mean': 0, 'comment_count': 0})
    merged_df['llm_mean_lag'] = merged_df['llm_mean'].shift(lag_days).fillna(0)
    
    return merged_df, filtered_comments


def plot_sentiment_pie(sentiment_counts):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4caf50' if lbl == '积极' else '#ff9800' if lbl == '中性' else '#f44336' for lbl in sentiment_counts.index]
    explode = [0.1 if lbl in ['积极', '消极'] else 0 for lbl in sentiment_counts.index]
    
    # 改为接收3个值（忽略texts）
    patches, texts, _ = ax.pie(
        sentiment_counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        explode=explode,
        autopct='%1.1f%%'
    )
    ax.set_title('LLM法情感标签分布', fontsize=14, fontproperties=font_prop)
    ax.axis('equal')
    return fig


# 页面主体
st.title('创业板个股股吧情绪对次日收益率的影响研究')

# 侧边栏（简化参数）
st.sidebar.subheader('股票选择')
stock_code = st.sidebar.selectbox('股票代码', ['300059'], index=0)
lag_days = st.sidebar.slider('情感滞后天数', 1, 3, 1)


# 加载数据并展示
try:
    comments_df, price_df = load_data(stock_code)
    merged_df, filtered_comments = process_data(comments_df, price_df, lag_days)
    total_count = len(filtered_comments)
    sentiment_counts = filtered_comments['llm_sentiment_label'].value_counts()
    
    # 1. 数据概览
    st.subheader('数据概览')
    st.write(f'- 有效评论数：{total_count} 条')
    st.write(f'- 交易日数量：{len(merged_df)} 个')
    
    # 2. 情感分布
    st.subheader('情感标签分布')
    pie_fig = plot_sentiment_pie(sentiment_counts)
    st.pyplot(pie_fig)
    
    # 3. 情感得分分布
    st.subheader('情感得分分布')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_comments['llm_sentiment_score'], bins=20, kde=True, ax=ax)
    ax.set_xlabel('情感得分')
    ax.set_ylabel('评论数量')
    st.pyplot(fig)
    
    # 4. 情感与收益率关系
    st.subheader('前1日情感得分与次日收益率')
    valid_data = merged_df[merged_df['llm_mean_lag'] != 0]
    if len(valid_data) >= 2:
        X = valid_data[['llm_mean_lag']].values
        y = valid_data['next_day_return'].values
        model = LinearRegression()
        model.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, y, alpha=0.6)
        ax.plot(X, model.predict(X), color='red', label=f'回归线 (R²={model.score(X, y):.4f})')
        ax.set_xlabel(f'前{lag_days}日情感得分')
        ax.set_ylabel('次日收益率（%）')
        ax.legend()
        st.pyplot(fig)
        
        st.write(f'- 回归R²：{model.score(X, y):.4f}')
        st.write(f'- 情感系数：{model.coef_[0]:.6f}')
    else:
        st.write('有效数据不足，无法绘制回归图')
    
    # 5. 评论示例
    st.subheader('评论示例')
    st.dataframe(filtered_comments[['post_publish_time', 'combined_text', 'llm_sentiment_label']].sample(5))

except Exception as e:
    st.error(f'错误：{str(e)}')
