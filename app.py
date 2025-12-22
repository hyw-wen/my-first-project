import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
import os
from matplotlib.font_manager import FontProperties

# 字体配置（修复乱码）
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
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        font_prop = FontProperties(family='SimHei')

setup_chinese_font()
warnings.filterwarnings('ignore')


# -------------------------- 核心：统一加载三种情感分析数据 --------------------------
@st.cache_data
def load_data(stock_code):
    # 加载包含三种方法的improved文件
    improved_file = f"{stock_code}_sentiment_analysis_improved_sentiment_analysis.csv"
    if not os.path.exists(improved_file):
        st.error(f"未找到文件：{improved_file}")
        st.stop()
    comments_df = pd.read_csv(improved_file)
    # 处理时间（论文范围：2025-11-22至2025-12-14）
    comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
    comments_df = comments_df[(comments_df['post_publish_time'] >= '2025-11-22') & 
                             (comments_df['post_publish_time'] <= '2025-12-14')]
    # 过滤空文本
    comments_df = comments_df[comments_df['combined_text'].notna() & (comments_df['combined_text'].str.strip() != '')]
    
    # 加载价格数据
    price_df = pd.read_csv(f"{stock_code}_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    price_df = price_df[(price_df['trade_date'] >= '2025-11-22') & 
                        (price_df['trade_date'] <= '2025-12-14')]
    price_df['next_day_return'] = price_df['next_day_return'].fillna(0)
    
    return comments_df, price_df


# -------------------------- 处理三种情感分析方法的数据 --------------------------
def process_data(comments_df, price_df, method="LLM法", lag_days=1):
    filtered_comments = comments_df.copy()
    # 匹配三种方法的字段
    if method == "词典法":
        score_col = "lexicon_sentiment"
        label_col = "lexicon_sentiment_label"  # 若文件中无此列，自动生成
    elif method == "LLM法":
        score_col = "llm_sentiment_score_new"
        label_col = "llm_sentiment_label_new"
    elif method == "集成法":
        score_col = "ensemble_sentiment_score_new"
        label_col = "ensemble_sentiment_label_new"  # 若文件中无此列，自动生成
    else:
        method = "LLM法"
        score_col = "llm_sentiment_score_new"
        label_col = "llm_sentiment_label_new"
    
    # 自动生成情感标签（若文件中无对应label列）
    if label_col not in filtered_comments.columns:
        def get_label(score):
            if score > 0.05:
                return "积极"
            elif score < -0.05:
                return "消极"
            else:
                return "中性"
        filtered_comments[label_col] = filtered_comments[score_col].apply(get_label)
    
    # 按日期聚合
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg({
        score_col: ["mean", "std", "count"],
        label_col: lambda x: x.value_counts()
    }).reset_index()
    daily_sentiment.columns = ["date", "mean_score", "std_score", "comment_count", "sentiment_dist"]
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    
    # 合并价格数据
    merged_df = pd.merge(price_df, daily_sentiment, left_on="trade_date", right_on="date", how="left")
    merged_df = merged_df.fillna({"mean_score": 0, "std_score": 0, "comment_count": 0})
    merged_df["mean_score_lag"] = merged_df["mean_score"].shift(lag_days).fillna(0)
    
    return merged_df, filtered_comments, score_col, label_col


# -------------------------- 绘制三种方法的对比表格 --------------------------
def plot_sentiment_table(comments_df):
    # 计算三种方法的统计结果（匹配论文表格）
    stats_data = []
    # 1. 词典法
    lexicon_mean = comments_df["lexicon_sentiment"].mean()
    lexicon_std = comments_df["lexicon_sentiment"].std()
    lexicon_pos = (comments_df["lexicon_sentiment"] > 0.05).sum() / len(comments_df) * 100
    lexicon_neu = ((comments_df["lexicon_sentiment"] >= -0.05) & (comments_df["lexicon_sentiment"] <= 0.05)).sum() / len(comments_df) * 100
    lexicon_neg = (comments_df["lexicon_sentiment"] < -0.05).sum() / len(comments_df) * 100
    stats_data.append(["词典法", f"{lexicon_mean:.3f}", f"{lexicon_std:.3f}", f"{lexicon_pos:.2f}%", f"{lexicon_neu:.2f}%", f"{lexicon_neg:.2f}%"])
    
    # 2. LLM法
    llm_mean = comments_df["llm_sentiment_score_new"].mean()
    llm_std = comments_df["llm_sentiment_score_new"].std()
    llm_pos = (comments_df["llm_sentiment_label_new"] == "积极").sum() / len(comments_df) * 100
    llm_neu = (comments_df["llm_sentiment_label_new"] == "中性").sum() / len(comments_df) * 100
    llm_neg = (comments_df["llm_sentiment_label_new"] == "消极").sum() / len(comments_df) * 100
    stats_data.append(["LLM法", f"{llm_mean:.3f}", f"{llm_std:.3f}", f"{llm_pos:.2f}%", f"{llm_neu:.2f}%", f"{llm_neg:.2f}%"])
    
    # 3. 集成法
    ensemble_mean = comments_df["ensemble_sentiment_score_new"].mean()
    ensemble_std = comments_df["ensemble_sentiment_score_new"].std()
    ensemble_pos = (comments_df["ensemble_sentiment_label_new"] == "积极").sum() / len(comments_df) * 100 if "ensemble_sentiment_label_new" in comments_df.columns else "-"
    ensemble_neu = (comments_df["ensemble_sentiment_label_new"] == "中性").sum() / len(comments_df) * 100 if "ensemble_sentiment_label_new" in comments_df.columns else "-"
    ensemble_neg = (comments_df["ensemble_sentiment_label_new"] == "消极").sum() / len(comments_df) * 100 if "ensemble_sentiment_label_new" in comments_df.columns else "-"
    stats_data.append(["集成法", f"{ensemble_mean:.3f}", f"{ensemble_std:.3f}", f"{ensemble_pos:.2f}%" if ensemble_pos != "-" else "-", f"{ensemble_neu:.2f}%" if ensemble_neu != "-" else "-", f"{ensemble_neg:.2f}%" if ensemble_neg != "-" else "-"])
    
    # 转为DataFrame展示
    stats_df = pd.DataFrame(stats_data, columns=["方法", "平均情感得分", "标准差", "积极比例", "中性比例", "消极比例"])
    return stats_df


# -------------------------- 页面主体：增加方法选择 --------------------------
st.title('创业板个股股吧情绪对次日收益率的影响研究')

# 侧边栏：增加“情感分析方法”选择
st.sidebar.subheader('分析配置')
stock_code = st.sidebar.selectbox('股票代码', ['300059'], index=0)
analysis_method = st.sidebar.selectbox('情感分析方法', ["LLM法", "词典法", "集成法"], index=0)
lag_days = st.sidebar.slider('情感滞后天数', 1, 3, 1)


# -------------------------- 加载数据并展示三种方法的结果 --------------------------
try:
    comments_df, price_df = load_data(stock_code)
    merged_df, filtered_comments, score_col, label_col = process_data(comments_df, price_df, analysis_method, lag_days)
    total_count = len(filtered_comments)
    sentiment_counts = filtered_comments[label_col].value_counts()
    
    # 1. 三种方法的对比表格（匹配论文）
    st.subheader('情感分析方法对比（论文表4）')
    stats_table = plot_sentiment_table(comments_df)
    st.dataframe(stats_table, use_container_width=True)
    
    # 2. 当前选择方法的情感分布
    st.subheader(f'{analysis_method}情感分析结果')
    col1, col2 = st.columns(2)
    with col1:
        st.write('### 情感标签分布')
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#4caf50' if lbl == '积极' else '#ff9800' if lbl == '中性' else '#f44336' for lbl in sentiment_counts.index]
        explode = [0.1 if lbl in ['积极', '消极'] else 0 for lbl in sentiment_counts.index]
        patches, texts, autotexts = ax.pie(
            sentiment_counts.values,
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            explode=explode,
            autopct='%1.1f%%',
            textprops={'fontproperties': font_prop}
        )
        for autotext in autotexts:
            autotext.set_color('white')
        ax.set_title(f'{analysis_method}情感标签分布', fontproperties=font_prop)
        ax.legend(sentiment_counts.index, prop=font_prop)
        st.pyplot(fig)
    
    with col2:
        st.write('### 情感得分分布')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_comments[score_col], bins=20, kde=True, ax=ax, color='#1f77b4', edgecolor='white')
        ax.axvline(0, color='orange', linestyle='--', label='中性线')
        ax.set_title(f'{analysis_method}情感得分分布', fontproperties=font_prop)
        ax.set_xlabel('情感得分', fontproperties=font_prop)
        ax.set_ylabel('评论数量', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        st.pyplot(fig)
    
    # 3. 情感与收益率回归
    st.subheader(f'{analysis_method}情感与次日收益率关系')
    valid_data = merged_df[merged_df['mean_score_lag'] != 0]
    if len(valid_data) >= 2:
        X = valid_data[['mean_score_lag']].values
        y = valid_data['next_day_return'].values
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, y, alpha=0.6, color='#1f77b4')
        ax.plot(X, model.predict(X), color='red', linewidth=2, label=f'回归线 (R²={r2:.4f})')
        ax.set_title(f'前{lag_days}日{analysis_method}情感得分与次日收益率', fontproperties=font_prop)
        ax.set_xlabel(f'前{lag_days}日情感得分', fontproperties=font_prop)
        ax.set_ylabel('次日收益率（%）', fontproperties=font_prop)
        ax.legend(prop=font_prop)
        st.pyplot(fig)
        st.write(f'回归R²：{r2:.4f} | 情感系数：{model.coef_[0]:.6f}')
    else:
        st.warning("有效样本不足，无法绘制回归图")
    
    # 4. 评论示例
    st.subheader('评论示例')
    sample_comments = filtered_comments[filtered_comments['combined_text'].notna()]
    if len(sample_comments) > 0:
        st.dataframe(sample_comments[['post_publish_time', 'combined_text', label_col, score_col]].sample(min(5, len(sample_comments))), use_container_width=True)
    else:
        st.write('暂无有效评论示例')

except Exception as e:
    st.error(f'运行错误：{str(e)}')
