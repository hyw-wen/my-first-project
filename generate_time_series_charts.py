import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
print("加载数据...")
try:
    df = pd.read_csv('300059_sentiment_analysis_updated.csv')
    price_df = pd.read_csv('300059_price_data.csv')
    print(f"成功加载数据：评论数据 {len(df)} 条，价格数据 {len(price_df)} 条")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# 数据预处理
df['post_publish_time'] = pd.to_datetime(df['post_publish_time'])
price_df['date'] = pd.to_datetime(price_df['trade_date'])

# 提取日期部分
df['date'] = df['post_publish_time'].dt.date
df['date'] = pd.to_datetime(df['date'])

# 按日期聚合情感数据
daily_sentiment = df.groupby('date').agg({
    'llm_sentiment_score_new': 'mean',
    'post_id': 'count'
}).rename(columns={'post_id': 'comment_count'})

# 合并情感数据和价格数据
merged_data = pd.merge(daily_sentiment, price_df, on='date', how='inner')
merged_data['daily_return'] = merged_data['pct_chg'] / 100  # 转换为小数

# 1. 时间序列分析图表
print("生成时间序列分析图表...")
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# 1.1 股价与收益率时间序列
ax1 = axes[0]
ax1_twin = ax1.twinx()

# 绘制收盘价
line1 = ax1.plot(merged_data['date'], merged_data['close'], 'b-', label='收盘价', linewidth=2)
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('收盘价 (元)', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 绘制收益率
line2 = ax1_twin.plot(merged_data['date'], merged_data['daily_return'], 'r-', label='日收益率', linewidth=1, alpha=0.7)
ax1_twin.set_ylabel('日收益率', fontsize=12, color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')

# 添加图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.set_title('股价与收益率时间序列', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 1.2 情感得分时间序列
ax2 = axes[1]
ax2.plot(merged_data['date'], merged_data['llm_sentiment_score_new'], 'g-', linewidth=2)
ax2.set_xlabel('日期', fontsize=12)
ax2.set_ylabel('平均情感得分', fontsize=12)
ax2.set_title('情感得分时间序列', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='中性线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 1.3 评论数量时间序列
ax3 = axes[2]
ax3.bar(merged_data['date'], merged_data['comment_count'], color='orange', alpha=0.7)
ax3.set_xlabel('日期', fontsize=12)
ax3.set_ylabel('评论数量', fontsize=12)
ax3.set_title('评论数量时间序列', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/时间序列分析图表.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 情感与收益率对比分析
print("生成情感与收益率对比分析图表...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 2.1 情感得分与收益率双轴图
ax1_twin = ax1.twinx()

# 绘制情感得分
line1 = ax1.plot(merged_data['date'], merged_data['llm_sentiment_score_new'], 'b-', label='情感得分', linewidth=2)
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('情感得分', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 绘制收益率
line2 = ax1_twin.plot(merged_data['date'], merged_data['daily_return'], 'r-', label='收益率', linewidth=1, alpha=0.7)
ax1_twin.set_ylabel('日收益率', fontsize=12, color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 添加图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.set_title('情感得分与收益率对比', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2.2 滞后效应分析
# 创建滞后数据
for lag in range(1, 6):
    merged_data[f'sentiment_lag_{lag}'] = merged_data['llm_sentiment_score_new'].shift(lag)

# 计算不同滞后期的相关性
lag_correlations = []
for lag in range(0, 6):
    if lag == 0:
        corr = merged_data['llm_sentiment_score_new'].corr(merged_data['daily_return'])
    else:
        corr = merged_data[f'sentiment_lag_{lag}'].corr(merged_data['daily_return'])
    lag_correlations.append(corr)

# 绘制滞后效应图
lags = list(range(0, 6))
bars = ax2.bar(lags, lag_correlations, color='skyblue', alpha=0.7)
ax2.set_xlabel('滞后天数', fontsize=12)
ax2.set_ylabel('相关系数', fontsize=12)
ax2.set_title('情感得分与收益率的滞后相关性', fontsize=14, fontweight='bold')
ax2.set_xticks(lags)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar, corr in zip(bars, lag_correlations):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
             f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

plt.tight_layout()
plt.savefig('final_visualizations/情感与收益率对比分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 情感与收益率散点图及回归线
print("生成情感与收益率散点图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 3.1 当日情感与当日收益率散点图
x = merged_data['llm_sentiment_score_new']
y = merged_data['daily_return']

# 计算回归线
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
x_pred = np.linspace(min(x), max(x), 100)
y_pred = intercept + slope * x_pred

# 计算置信区间
n = len(x)
t = stats.t.ppf(0.975, n-2)  # 95%置信区间
std_err_pred = std_err * np.sqrt(1/n + (x_pred - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
ci_low = y_pred - t * std_err_pred
ci_high = y_pred + t * std_err_pred

# 绘制散点图和回归线
ax1.scatter(x, y, alpha=0.6, color='blue')
ax1.plot(x_pred, y_pred, color='red', linewidth=2, label=f'回归线 (R²={r_value**2:.3f})')
ax1.fill_between(x_pred, ci_low, ci_high, color='red', alpha=0.2, label='95%置信区间')
ax1.set_xlabel('情感得分', fontsize=12)
ax1.set_ylabel('日收益率', fontsize=12)
ax1.set_title('当日情感得分与日收益率关系', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 滞后一天情感与当日收益率散点图
x_lag = merged_data['sentiment_lag_1'].dropna()
y_lag = merged_data.loc[x_lag.index, 'daily_return']

# 计算回归线
slope_lag, intercept_lag, r_value_lag, p_value_lag, std_err_lag = stats.linregress(x_lag, y_lag)
x_pred_lag = np.linspace(min(x_lag), max(x_lag), 100)
y_pred_lag = intercept_lag + slope_lag * x_pred_lag

# 计算置信区间
n_lag = len(x_lag)
t_lag = stats.t.ppf(0.975, n_lag-2)  # 95%置信区间
std_err_pred_lag = std_err_lag * np.sqrt(1/n_lag + (x_pred_lag - np.mean(x_lag))**2 / np.sum((x_lag - np.mean(x_lag))**2))
ci_low_lag = y_pred_lag - t_lag * std_err_pred_lag
ci_high_lag = y_pred_lag + t_lag * std_err_pred_lag

# 绘制散点图和回归线
ax2.scatter(x_lag, y_lag, alpha=0.6, color='green')
ax2.plot(x_pred_lag, y_pred_lag, color='red', linewidth=2, label=f'回归线 (R²={r_value_lag**2:.3f})')
ax2.fill_between(x_pred_lag, ci_low_lag, ci_high_lag, color='red', alpha=0.2, label='95%置信区间')
ax2.set_xlabel('前一日情感得分', fontsize=12)
ax2.set_ylabel('当日收益率', fontsize=12)
ax2.set_title('前一日情感得分与当日收益率关系', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/情感与收益率散点图.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 移动平均情感分析
print("生成移动平均情感分析图表...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 4.1 不同窗口的移动平均情感
for window in [3, 5, 7]:
    merged_data[f'sentiment_ma_{window}'] = merged_data['llm_sentiment_score_new'].rolling(window=window).mean()
    ax1.plot(merged_data['date'], merged_data[f'sentiment_ma_{window}'], 
             label=f'{window}日移动平均', linewidth=2, alpha=0.8)

ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('移动平均情感得分', fontsize=12)
ax1.set_title('不同窗口的移动平均情感得分', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='中性线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4.2 移动平均情感与收益率对比
ax2_twin = ax2.twinx()

# 绘制5日移动平均情感
line1 = ax2.plot(merged_data['date'], merged_data['sentiment_ma_5'], 'b-', label='5日移动平均情感', linewidth=2)
ax2.set_xlabel('日期', fontsize=12)
ax2.set_ylabel('移动平均情感得分', fontsize=12, color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 绘制收益率
line2 = ax2_twin.plot(merged_data['date'], merged_data['daily_return'], 'r-', label='收益率', linewidth=1, alpha=0.7)
ax2_twin.set_ylabel('日收益率', fontsize=12, color='r')
ax2_twin.tick_params(axis='y', labelcolor='r')
ax2_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 添加图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left')
ax2.set_title('5日移动平均情感与收益率对比', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/移动平均情感分析.png', dpi=300, bbox_inches='tight')
plt.close()

print("时间序列分析图表生成完成！")
