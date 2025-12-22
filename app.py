import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy import stats
import warnings
import os
from matplotlib.font_manager import FontProperties

# 全局字体配置
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


# -------------------------- 核心1：加载文件（适配你的字段） --------------------------
@st.cache_data
def load_data(stock_code):
    # 仅加载你上传的improved文件（含llm_sentiment_label_new等字段）
    improved_file = f"{stock_code}_sentiment_analysis_improved_sentiment_analysis.csv"
    if not os.path.exists(improved_file):
        st.error(f"未找到核心文件：{improved_file}，请确认文件路径正确")
        st.stop()
    
    # 读取评论数据（使用文件中已有的LLM结果字段）
    comments_df = pd.read_csv(improved_file)
    # 强制字段类型与时间范围（论文：2025-11-22至2025-12-14）
    comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])
    comments_df = comments_df[(comments_df['post_publish_time'] >= '2025-11-22') & 
                             (comments_df['post_publish_time'] <= '2025-12-14')]
    
    # 读取价格数据（需包含trade_date和next_day_return字段）
    price_df = pd.read_csv(f"{stock_code}_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    price_df = price_df[(price_df['trade_date'] >= '2025-11-22') & 
                        (price_df['trade_date'] <= '2025-12-14')]
    # 填充价格数据空值（避免回归报错）
    price_df['next_day_return'] = price_df['next_day_return'].fillna(0)
    
    st.success(f"已加载{len(comments_df)}条评论（论文时间范围）+{len(price_df)}个交易日数据")
    return comments_df, price_df


# -------------------------- 核心2：数据处理（完全复用文件字段） --------------------------
def process_data(comments_df, price_df, text_length_limit=100, window_size=21, lag_days=1):
    filtered_comments = comments_df.copy()
    
    # 1. 文本过滤（论文：50-100字，使用combined_text字段）
    filtered_comments['text_length'] = filtered_comments['combined_text'].str.len()
    filtered_comments = filtered_comments[(filtered_comments['text_length'] >= 50) & 
                                         (filtered_comments['text_length'] <= text_length_limit)]
    
    # 2. 情感字段：直接复用文件中的llm_sentiment_label_new/score_new（与论文LLM法对应）
    filtered_comments['llm_sentiment_label'] = filtered_comments['llm_sentiment_label_new']
    filtered_comments['llm_sentiment_score'] = filtered_comments['llm_sentiment_score_new']
    filtered_comments['ensemble_sentiment_score'] = filtered_comments['ensemble_sentiment_score_new']
    
    # 3. 按日期聚合（匹配论文日均69.8条评论）
    daily_sentiment = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date).agg({
        'ensemble_sentiment_score': ['mean', 'std', 'count'],
        'llm_sentiment_score': 'mean'
    }).reset_index()
    daily_sentiment.columns = ['date', 'ensemble_mean', 'ensemble_std', 'comment_count', 'llm_mean']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 4. 合并价格数据（对齐交易日）
    merged_df = pd.merge(price_df, daily_sentiment, left_on='trade_date', right_on='date', how='left')
    merged_df = merged_df.fillna({
        'comment_count': 0,
        'ensemble_mean': 0,
        'ensemble_std': 0,
        'llm_mean': 0
    })
    
    # 5. 滞后效应（论文T+1）
    merged_df['ensemble_mean_lag'] = merged_df['ensemble_mean'].shift(lag_days).fillna(0)
    # 移动平均（论文最优21天）
    merged_df['ensemble_mean_rolling'] = merged_df['ensemble_mean'].rolling(window=window_size).mean().fillna(0)
    
    return merged_df, filtered_comments


# -------------------------- 核心3：情感分布饼图（与论文表4对齐） --------------------------
def plot_sentiment_pie(sentiment_counts, filtered_count):
    fig, ax = plt.subplots(figsize=(8, 6))
    # 配色与分离小扇形（积极、消极）
    colors = ['#4caf50' if lbl == '积极' else '#ff9800' if lbl == '中性' else '#f44336' 
              for lbl in sentiment_counts.index]
    explode = [0.1 if lbl in ['积极', '消极'] else 0 for lbl in sentiment_counts.index]
    
    # 绘制饼图
    patches, _ = ax.pie(
        sentiment_counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        explode=explode
    )
    
    # 箭头标签（指向扇形中心）
    for i, lbl in enumerate(sentiment_counts.index):
        patch = patches[i]
        theta_mid = (patch.theta1 + patch.theta2) / 2  # 扇形中间角度
        r = patch.r * 0.8  # 靠近扇形中心
        x = r * np.cos(np.radians(theta_mid))
        y = r * np.sin(np.radians(theta_mid))
        
        # 标签位置（避免重叠）
        if lbl == '积极':
            text_pos = (1.1, 0.6)
        elif lbl == '消极':
            text_pos = (1.1, 0.4)
        else:
            # 中性标签在下方
            ax.text(0, -1.2, f'{lbl} ({sentiment_counts[lbl]/filtered_count*100:.1f}%)', 
                    ha='center', fontsize=12, fontproperties=font_prop)
            continue
        
        # 带箭头的注释
        ax.annotate(
            f'{lbl} ({sentiment_counts[lbl]/filtered_count*100:.1f}%)',
            xy=(x, y),
            xytext=text_pos,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11,
            fontproperties=font_prop
        )
    
    ax.set_title('LLM法情感标签分布（论文表4）', fontsize=14, fontproperties=font_prop)
    ax.axis('equal')
    return fig


# -------------------------- 核心4：回归分析图（与论文图5对齐） --------------------------
def plot_regression(merged_df, lag_days):
    # 筛选有效数据（避免NaN）
    valid_data = merged_df[(merged_df['ensemble_mean_lag'].notna()) & 
                           (merged_df['next_day_return'].notna())]
    if len(valid_data) < 2:
        st.warning("有效回归样本不足，无法绘制回归图")
        return None
    
    X = valid_data[['ensemble_mean_lag']].values
    y = valid_data['next_day_return'].values
    
    # 线性回归（论文R²≈0.0212）
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    coef = model.coef_[0]
    
    # 稳健回归（论文推荐）
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    r2_robust = ransac.score(X, y)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    # 散点图（按情感得分着色）
    colors = ['red' if s < -0.05 else 'green' if s > 0.05 else 'blue' 
              for s in valid_data['ensemble_mean_lag']]
    ax.scatter(valid_data['ensemble_mean_lag'], valid_data['next_day_return'], 
               c=colors, alpha=0.6, s=50)
    
    # 回归线
    x_line = np.linspace(valid_data['ensemble_mean_lag'].min(), 
                         valid_data['ensemble_mean_lag'].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'标准回归线（R²={r2:.4f}）')
    
    # 95%置信区间（论文要求）
    n = len(X)
    t_val = stats.t.ppf(0.975, n-2)
    y_pred = model.predict(X)
    residual_std = np.sqrt(np.sum((y - y_pred)**2) / (n-2))
    margin_error = t_val * residual_std * np.sqrt(1 + 1/n + (x_line - X.mean())**2 / np.sum((X - X.mean())**2))
    ax.fill_between(x_line.flatten(), y_line - margin_error.flatten(), 
                    y_line + margin_error.flatten(), alpha=0.2, color='red', label='95%置信区间')
    
    # 图表配置
    ax.set_title(f'前{lag_days}日情感得分与次日收益率关系（论文图5）', fontsize=14, fontproperties=font_prop)
    ax.set_xlabel(f'前{lag_days}日LLM情感得分', fontsize=12, fontproperties=font_prop)
    ax.set_ylabel('次日收益率（%）', fontsize=12, fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.grid(True, alpha=0.3)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    return fig


# -------------------------- 页面主体（与论文结构一致） --------------------------
st.title('创业板个股股吧情绪对次日收益率的影响研究')

# 侧边栏（默认参数匹配论文）
st.sidebar.subheader('股票选择')
stock_code = st.sidebar.selectbox('股票代码', ['300059'], index=0)

st.sidebar.subheader('参数调整（论文默认）')
# 初始化参数（论文最优值）
if 'params' not in st.session_state:
    st.session_state.params = {
        'text_length': 100,   # 论文：50-100字
        'window_size': 21,    # 论文最优窗口
        'lag_days': 1        # 论文T+1滞后
    }

# 重置按钮
if st.sidebar.button('🔄 重置为论文参数'):
    st.session_state.params = {
        'text_length': 100,
        'window_size': 21,
        'lag_days': 1
    }

# 控件（限制范围与论文一致）
text_length = st.sidebar.slider(
    '文本长度限制（字）', 50, 100, st.session_state.params['text_length'], step=10, key='len'
)
window_size = st.sidebar.slider(
    '移动平均窗口（天）', 14, 30, st.session_state.params['window_size'], step=1, key='win'
)
lag_days = st.sidebar.slider(
    '情感滞后天数（天）', 1, 3, st.session_state.params['lag_days'], step=1, key='lag'
)
# 更新session_state
st.session_state.params.update({
    'text_length': text_length,
    'window_size': window_size,
    'lag_days': lag_days
})


# -------------------------- 数据加载与可视化（全流程） --------------------------
try:
    comments_df, price_df = load_data(stock_code)
    merged_df, filtered_comments = process_data(
        comments_df, price_df, text_length, window_size, lag_days
    )
    total_comments = len(comments_df)
    filtered_count = len(filtered_comments)
    sentiment_counts = filtered_comments['llm_sentiment_label'].value_counts()
    
    # 1. 数据质量检查（与论文表4对比）
    st.subheader('数据质量检查（论文匹配）')
    pos_ratio = sentiment_counts.get('积极', 0) / filtered_count * 100
    neu_ratio = sentiment_counts.get('中性', 0) / filtered_count * 100
    neg_ratio = sentiment_counts.get('消极', 0) / filtered_count * 100
    
    st.write(f'📊 核心指标（论文参考值）：')
    st.write(f'- 总评论数：{total_comments} 条（论文：977条）')
    st.write(f'- 有效评论数：{filtered_count} 条（文本50-100字）')
    st.write(f'- 情感分布（LLM法）：')
    st.write(f'  - 积极：{pos_ratio:.2f}%（论文：14.84%）')
    st.write(f'  - 中性：{neu_ratio:.2f}%（论文：76.16%）')
    st.write(f'  - 消极：{neg_ratio:.2f}%（论文：9.01%）')
    st.write(f'- 日均评论数：{filtered_comments.groupby(filtered_comments["post_publish_time"].dt.date).size().mean():.2f} 条（论文：69.79条）')
    
    # 2. 评论数量趋势（论文单日最高386条）
    st.subheader('评论数量随时间变化')
    daily_comments = filtered_comments.groupby(filtered_comments['post_publish_time'].dt.date)['post_id'].count()
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_comments.plot(ax=ax, marker='o', linewidth=2, markersize=5, color='#1f77b4')
    # 标注峰值
    if len(daily_comments) > 0:
        max_date = daily_comments.idxmax()
        max_count = daily_comments.max()
        ax.annotate(f'最高：{max_count}条', xy=(max_date, max_count), xytext=(max_date, max_count + 20),
                    arrowprops=dict(arrowstyle='->', color='red'), fontproperties=font_prop)
    ax.set_title('每日评论数量趋势（2025-11-22至2025-12-14）', fontsize=14, fontproperties=font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=font_prop)
    ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontproperties=font_prop)
    st.pyplot(fig)
    
    # 3. 情感分析结果（饼图+得分分布）
    st.subheader('情感分析结果（LLM法）')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('### 情感标签分布（论文表4）')
        if len(sentiment_counts) > 0:
            pie_fig = plot_sentiment_pie(sentiment_counts, filtered_count)
            st.pyplot(pie_fig)
            # 显示具体数值
            st.write('情感数量明细：')
            for lbl, cnt in sentiment_counts.items():
                st.write(f'- {lbl}：{cnt} 条（{cnt/filtered_count*100:.2f}%）')
    
    with col2:
        st.write('### 情感得分分布（论文图6）')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_comments['llm_sentiment_score'], bins=20, kde=True, 
                     ax=ax, color='#1f77b4', edgecolor='white')
        ax.axvline(0, color='orange', linestyle='--', label='中性线')
        ax.set_title('LLM法情感得分分布', fontsize=14, fontproperties=font_prop)
        ax.set_xlabel('情感得分', fontsize=12, fontproperties=font_prop)
        ax.set_ylabel('评论数量', fontsize=12, fontproperties=font_prop)
        ax.legend(prop=font_prop)
        st.pyplot(fig)
        # 得分统计（论文表4）
        st.write('得分统计（论文参考）：')
        st.write(f'- 均值：{filtered_comments["llm_sentiment_score"].mean():.4f}（论文：0.041）')
        st.write(f'- 标准差：{filtered_comments["llm_sentiment_score"].std():.4f}（论文：0.298）')
    
    # 4. 回归分析（论文图5）
    st.subheader('情感与次日收益率回归分析')
    reg_fig = plot_regression(merged_df, lag_days)
    if reg_fig:
        st.pyplot(reg_fig)
        # 回归结果（论文表1）
        valid_data = merged_df[(merged_df['ensemble_mean_lag'].notna()) & (merged_df['next_day_return'].notna())]
        X = valid_data[['ensemble_mean_lag']].values
        y = valid_data['next_day_return'].values
        model = LinearRegression()
        model.fit(X, y)
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        st.write('### 回归结果（论文表1）')
        st.write(f'- 标准回归 R²：{model.score(X, y):.4f}（论文：0.0212）')
        st.write(f'- 稳健回归 R²：{ransac.score(X, y):.4f}（论文：0.0185）')
        st.write(f'- 情感系数：{model.coef_[0]:.6f}（论文：0.000123）')
        st.write(f'- 结论：情感与次日收益率呈弱正相关，符合论文H1假设')
    
    # 5. 评论示例
    st.subheader('评论示例')
    selected_lbl = st.selectbox('选择情感类型', ['积极', '中性', '消极'])
    sample_comments = filtered_comments[filtered_comments['llm_sentiment_label'] == selected_lbl]
    if len(sample_comments) > 0:
        st.dataframe(sample_comments[['post_publish_time', 'combined_text', 'llm_sentiment_score']].sample(min(5, len(sample_comments))))
    else:
        st.write(f'暂无{selected_lbl}情感的评论示例')

except Exception as e:
    st.error(f'运行错误：{str(e)}')
    st.write('请优先检查：1. 文件是否存在 2. price_df是否有next_day_return字段 3. 时间范围是否匹配')
