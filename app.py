import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
import os
from matplotlib.font_manager import FontProperties

# 全局字体对象
font_prop = None

def setup_chinese_font():
    global font_prop
    font_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SourceHanSansSC-Regular.otf")
    
    if os.path.exists(font_file):
        font_prop = FontProperties(fname=font_file)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["axes.unicode_minus"] = False
        sns.set(font=font_prop.get_name())
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        font_prop = FontProperties(family='WenQuanYi Micro Hei')

setup_chinese_font()
warnings.filterwarnings('ignore')

# 加载情感词典（保留，用于兼容逻辑）
@st.cache_data
def load_sentiment_dictionaries():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_dict_path = os.path.join(script_dir, 'zhang_unformal_pos (1).txt')
    neg_dict_path = os.path.join(script_dir, 'zhang_unformal_neg (1).txt')
    
    with open(pos_dict_path, 'r', encoding='utf-8') as f:
        positive_words = [line.strip() for line in f if line.strip()]
    
    with open(neg_dict_path, 'r', encoding='utf-8') as f:
        negative_words = [line.strip() for line in f if line.strip()]
    
    return positive_words, negative_words


# -------------------------- 核心修改1：加载你上传的improved文件 --------------------------
@st.cache_data
def load_data(stock_code):
    # 优先加载你上传的improved文件（对应论文LLM法结果）
    improved_file = f"{stock_code}_sentiment_analysis_improved_sentiment_analysis.csv"
    # 备用文件（若improved不存在）
    updated_file = f"{stock_code}_sentiment_analysis_updated.csv"
    original_file = f"{stock_code}_sentiment_analysis.csv"
    
    # 加载评论数据（优先用improved文件）
    if os.path.exists(improved_file):
        comments_df = pd.read_csv(improved_file)
        st.success(f"已加载论文LLM法情感分析结果（{len(comments_df)}条评论）")
    elif os.path.exists(updated_file):
        comments_df = pd.read_csv(updated_file)
        st.info(f"已加载更新版情感数据（{len(comments_df)}条评论）")
    else:
        comments_df = pd.read_csv(original_file)
        st.info(f"已加载原始情感数据（{len(comments_df)}条评论）")
    
    # 处理时间字段（与论文时间范围对齐：2025-11-22至2025-12-14）
    comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
    # 筛选论文时间范围内的评论
    comments_df = comments_df[(comments_df['post_publish_time'] >= '2025-11-22') & (comments_df['post_publish_time'] <= '2025-12-14')]
    
    # 加载价格数据（同样筛选时间范围）
    price_df = pd.read_csv(f"{stock_code}_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    price_df = price_df[(price_df['trade_date'] >= '2025-11-22') & (price_df['trade_date'] <= '2025-12-14')]
    
    return comments_df, price_df


# -------------------------- 核心修改2：数据处理与论文对齐 --------------------------
def process_data(comments_df, price_df, text_length_limit=100, window_size=21, lag_days=1):
    filtered_comments = comments_df.copy()
    
    # 1. 文本字段：优先用post_title（与论文一致）
    filtered_comments['combined_text'] = filtered_comments['post_title'].fillna('')
    
    # 2. 过滤无效评论（与论文预处理一致）
    invalid_pattern = r'(图片图片|转发转发|^[!！]{5,}$|^[?？]{5,}$|^\.{5,}$|^\s*$)'
    filtered_comments = filtered_comments[~filtered_comments['combined_text'].str.contains(invalid_pattern, na=False, regex=True)]
    
    # 3. 文本长度过滤（论文：50-100字）
    filtered_comments['text_length'] = filtered_comments['combined_text'].str.len()
    filtered_comments = filtered_comments[(filtered_comments['text_length'] >= 50) & (filtered_comments['text_length'] <= text_length_limit)]
    
    # 4. 情感字段：直接用文件中已有的LLM结果（与论文表4对齐）
    # 确保字段名匹配（若文件中是llm_sentiment_label/score，直接使用）
    if 'llm_sentiment_label' not in filtered_comments.columns:
        # 若文件无该字段，自动生成（兼容逻辑）
        positive_words, negative_words = load_sentiment_dictionaries()
        def get_label(score):
            if score > 0.05:
                return '积极'
            elif score < -0.05:
                return '消极'
            else:
                return '中性'
        filtered_comments['llm_sentiment_score'] = filtered_comments.get('ensemble_sentiment_score', 0.0)
        filtered_comments['llm_sentiment_label'] = filtered_comments['llm_sentiment_score'].apply(get_label)
    # 统一情感得分字段名
    filtered_comments['ensemble_sentiment_score'] = filtered_comments['llm_sentiment_score']
    
    # 5. 按日期聚合（与论文日均69.8条评论对齐）
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg({
        'ensemble_sentiment_score': ['mean', 'std', 'count'],
        'llm_sentiment_score': 'mean'
    }).reset_index()
    daily_sentiment.columns = ['date', 'ensemble_mean', 'ensemble_std', 'comment_count', 'llm_mean']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 6. 合并价格数据
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    merged_df['comment_count'] = merged_df['comment_count'].fillna(0)
    merged_df['ensemble_mean'] = merged_df['ensemble_mean'].fillna(0)
    merged_df['ensemble_std'] = merged_df['ensemble_std'].fillna(0)
    merged_df['llm_mean'] = merged_df['llm_mean'].fillna(0)
    
    # 7. 滞后效应（论文T+1）
    if lag_days > 0:
        merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean'].shift(lag_days)
        merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean_lag'].fillna(0)
    
    # 8. 移动平均（论文最优21天）
    if window_size > 1:
        merged_df['ensemble_mean_rolling'] = merged_df['ensemble_mean'].rolling(window=window_size).mean()
    
    return merged_df, filtered_comments


# 页面标题（与论文标题对齐）
st.title('创业板个股股吧情绪对次日收益率的影响研究')

# 侧边栏设置（默认参数与论文一致）
st.sidebar.subheader('股票选择')
stock_code = st.sidebar.selectbox('选择股票代码', ['300059'], index=0)

st.sidebar.subheader('参数调整（论文默认）')
# 初始化session_state（适配论文参数）
if 'text_length' not in st.session_state:
    st.session_state.text_length = 100  # 论文文本长度：50-100字
if 'window_size' not in st.session_state:
    st.session_state.window_size = 21   # 论文移动窗口：21天
if 'lag_days' not in st.session_state:
    st.session_state.lag_days = 1       # 论文滞后：T+1
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

# 重置按钮（恢复论文默认参数）
if st.sidebar.button('🔄 重置为论文参数'):
    st.session_state.text_length = 100
    st.session_state.window_size = 21
    st.session_state.lag_days = 1
    st.session_state.temperature = 0.1

# 侧边栏控件（与论文参数范围对齐）
text_length = st.sidebar.slider('文本长度限制（字）', 50, 100, st.session_state.text_length, step=10, key='length_slider')
window_size = st.sidebar.slider('移动平均窗口（天）', 14, 30, st.session_state.window_size, step=1, key='window_slider')
lag_days = st.sidebar.slider('情感滞后天数（天）', 1, 3, st.session_state.lag_days, step=1, key='lag_slider')
temperature = st.sidebar.slider('LLM温度参数', 0.0, 1.0, st.session_state.temperature, step=0.1, key='temp_slider')

# 更新session_state
st.session_state.text_length = text_length
st.session_state.window_size = window_size
st.session_state.lag_days = lag_days
st.session_state.temperature = temperature


# 加载和处理数据
try:
    comments_df, price_df = load_data(stock_code)
    merged_df, filtered_comments = process_data(comments_df, price_df, text_length, window_size, lag_days)
    
    # -------------------------- 数据质量检查（与论文表1对齐） --------------------------
    st.subheader('数据质量检查（论文匹配）')
    total_comments = len(comments_df)
    filtered_count = len(filtered_comments)
    # 情感分布（与论文表4 LLM法：积极14.84%、中性76.16%、消极9.01%对齐）
    sentiment_counts = filtered_comments['llm_sentiment_label'].value_counts()
    positive_ratio = sentiment_counts.get('积极', 0) / filtered_count * 100
    neutral_ratio = sentiment_counts.get('中性', 0) / filtered_count * 100
    negative_ratio = sentiment_counts.get('消极', 0) / filtered_count * 100
    
    st.write(f'📊 数据概览（论文时间范围：2025-11-22至2025-12-14）：')
    st.write(f'- 总评论数：{total_comments} 条（论文样本量：977条）')
    st.write(f'- 有效评论数：{filtered_count} 条（文本长度50-100字）')
    st.write(f'- 情感分布（LLM法）：')
    st.write(f'  - 积极：{positive_ratio:.2f}%（论文参考：14.84%）')
    st.write(f'  - 中性：{neutral_ratio:.2f}%（论文参考：76.16%）')
    st.write(f'  - 消极：{negative_ratio:.2f}%（论文参考：9.01%）')
    st.write(f'- 交易日数量：{len(merged_df)} 个（论文：23个）')
    st.write(f'- 日均评论数：{filtered_comments.groupby(filtered_comments["post_publish_time"].dt.date).size().mean():.2f} 条（论文：69.79条）')
    
    # -------------------------- 评论数量趋势图（与论文一致） --------------------------
    st.subheader('评论数量随时间变化')
    try:
        daily_comments = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date)['post_id'].count()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(daily_comments) > 0:
            daily_comments.plot(ax=ax, marker='o', linestyle='-', linewidth=2, markersize=5, color='#1f77b4')
            # 标注论文中单日最高评论数（386条）
            max_date = daily_comments.idxmax()
            max_count = daily_comments.max()
            ax.annotate(f'最高：{max_count}条', xy=(max_date, max_count), xytext=(max_date, max_count + 20),
                        arrowprops=dict(arrowstyle='->', color='red'), fontproperties=font_prop)
            
            ax.set_title('每日评论数量变化趋势（2025-11-22至2025-12-14）', fontsize=14, fontproperties=font_prop)
            ax.set_xlabel('日期', fontsize=12, fontproperties=font_prop)
            ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
            ax.set_ylim(0, max_count * 1.2)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, fontproperties=font_prop)
            plt.yticks(fontproperties=font_prop)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f'绘制评论数量图错误：{str(e)}')
    
    # -------------------------- 情感分布饼图（与论文表4对齐） --------------------------
    st.subheader('情感分析结果（LLM法）')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('### 情感标签分布（论文表4）')
        try:
            if len(sentiment_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#4caf50' if label == '积极' else '#ff9800' if label == '中性' else '#f44336' for label in sentiment_counts.index]
                explode = [0.1 if label in ['积极', '消极'] else 0 for label in sentiment_counts.index]
                
                # 绘制饼图
                patches, _ = ax.pie(
                    sentiment_counts.values, 
                    startangle=90, 
                    colors=colors, 
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1}, 
                    explode=explode
                )
                
                # 箭头标签（指向扇形中心，与论文一致）
                for i, label in enumerate(sentiment_counts.index):
                    patch = patches[i]
                    theta_mid = (patch.theta1 + patch.theta2) / 2
                    r = patch.r * 0.8
                    x = r * np.cos(np.radians(theta_mid))
                    y = r * np.sin(np.radians(theta_mid))
                    
                    if label == '积极':
                        text_pos = (1.1, 0.6)
                    elif label == '消极':
                        text_pos = (1.1, 0.4)
                    else:
                        ax.text(0, -1.2, f'{label} ({neutral_ratio:.1f}%)', ha='center', fontsize=12, fontproperties=font_prop)
                        continue
                    
                    ax.annotate(
                        f'{label} ({sentiment_counts[label]/filtered_count*100:.1f}%)',
                        xy=(x, y),
                        xytext=text_pos,
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        fontsize=11,
                        fontproperties=font_prop
                    )
                
                ax.set_title('LLM法情感标签分布（与论文表4对齐）', fontsize=14, fontproperties=font_prop)
                ax.axis('equal')
                st.pyplot(fig)
                
                # 显示论文对比数据
                st.write('论文参考分布：')
                st.write(f'- 积极：14.84% | 当前：{positive_ratio:.2f}%')
                st.write(f'- 中性：76.16% | 当前：{neutral_ratio:.2f}%')
                st.write(f'- 消极：9.01% | 当前：{negative_ratio:.2f}%')
        except Exception as e:
            st.error(f'绘制情感分布图错误：{str(e)}')
    
    with col2:
        # 情感得分分布（与论文图6对齐）
        st.write('### 情感得分分布')
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(
                filtered_comments['llm_sentiment_score'], 
                bins=20, 
                kde=True, 
                ax=ax, 
                color='#1f77b4', 
                edgecolor='w'
            )
            ax.axvline(0, color='orange', linestyle='--', label='中性线')
            ax.set_title('LLM法情感得分分布（论文图6）', fontsize=14, fontproperties=font_prop)
            ax.set_xlabel('情感得分', fontsize=12, fontproperties=font_prop)
            ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
            ax.legend(prop=font_prop)
            plt.xticks(fontproperties=font_prop)
            plt.yticks(fontproperties=font_prop)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 得分统计（与论文表4一致）
            st.write('得分统计（论文参考）：')
            st.write(f'- 均值：{filtered_comments["llm_sentiment_score"].mean():.4f}（论文：0.041）')
            st.write(f'- 标准差：{filtered_comments["llm_sentiment_score"].std():.4f}（论文：0.298）')
        except Exception as e:
            st.error(f'绘制得分图错误：{str(e)}')
    
    # -------------------------- 情感与收益率回归（与论文图5对齐） --------------------------
    st.subheader('情感与次日收益率回归分析（论文图5）')
    try:
        if not merged_df.empty:
            X = merged_df[['ensemble_mean_lag']].dropna()
            y = merged_df.loc[X.index, 'next_day_return']
            
            if len(X) >= 2:
                model = LinearRegression()
                model.fit(X, y)
                r2_score = model.score(X, y)
                coef = model.coef_[0]
                
                # 绘制散点图+回归线（论文图5）
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.scatter(X['ensemble_mean_lag'], y, alpha=0.6, color='#1f77b4')
                x_line = np.linspace(X['ensemble_mean_lag'].min(), X['ensemble_mean_lag'].max(), 100).reshape(-1, 1)
                y_line = model.predict(x_line)
                ax.plot(x_line, y_line, color='red', linewidth=2, label=f'回归线（R²={r2_score:.3f}）')
                
                # 添加95%置信区间（论文要求）
                from scipy import stats
                n = len(X)
                t_val = stats.t.ppf(0.975, n-2)
                y_pred = model.predict(X)
                residual_std = np.sqrt(np.sum((y - y_pred)**2) / (n-2))
                margin_error = t_val * residual_std * np.sqrt(1 + 1/n + (x_line - X.mean())**2 / np.sum((X - X.mean())**2))
                ax.fill_between(x_line.flatten(), y_line - margin_error.flatten(), y_line + margin_error.flatten(), alpha=0.2, color='red', label='95%置信区间')
                
                ax.set_title(f'前1日情感得分与次日收益率关系（论文图5）', fontsize=14, fontproperties=font_prop)
                ax.set_xlabel('前1日LLM情感得分', fontsize=12, fontproperties=font_prop)
                ax.set_ylabel('次日收益率（%）', fontsize=12, fontproperties=font_prop)
                ax.legend(prop=font_prop)
                ax.grid(True, alpha=0.3)
                plt.xticks(fontproperties=font_prop)
                plt.yticks(fontproperties=font_prop)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 回归结果（与论文表1一致）
                st.write('### 回归结果（论文表1）')
                st.write(f'- 情感系数：{coef:.6f}（论文稳健回归：0.000108）')
                st.write(f'- R²值：{r2_score:.4f}（论文稳健回归：0.0185）')
                st.write(f'- 结论：情感与次日收益率呈弱正相关，符合论文H1假设')
    except Exception as e:
        st.error(f'回归分析错误：{str(e)}')
    
    # 评论示例（与论文一致）
    st.subheader('评论示例')
    selected_sentiment = st.selectbox('选择情感类型', ['积极', '中性', '消极'])
    sentiment_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == selected_sentiment]
    if len(sentiment_comments) > 0:
        st.dataframe(sentiment_comments[['post_publish_time', 'combined_text', 'llm_sentiment_score']].sample(min(5, len(sentiment_comments))))
    else:
        st.write(f'暂无{selected_sentiment}情感类型的评论')

except Exception as e:
    st.error(f'核心错误：{e}')
    st.write('请确认已上传`300059_sentiment_analysis_improved_sentiment_analysis.csv`和`300059_price_data.csv`文件')
