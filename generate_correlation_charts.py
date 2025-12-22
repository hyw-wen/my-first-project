import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
print("加载数据...")
df = pd.read_csv('300059_sentiment_analysis_updated.csv')
price_df = pd.read_csv('300059_price_data.csv')

# 转换日期格式
df['date'] = pd.to_datetime(df['post_publish_time']).dt.date  # 只保留日期部分
price_df['trade_date'] = pd.to_datetime(price_df['trade_date']).dt.date  # 只保留日期部分

# 按日期合并数据
merged_data = pd.merge(df, price_df, left_on='date', right_on='trade_date', how='inner')

# 计算日收益率
merged_data['daily_return'] = merged_data['pct_chg']  # 使用已有的涨跌幅数据

# 计算前一日情感得分
merged_data['prev_sentiment'] = merged_data['llm_sentiment_score_new'].shift(1)

# 删除缺失值
merged_data = merged_data.dropna()

print(f"成功加载数据：合并数据 {len(merged_data)} 条")

# 创建输出目录
output_dir = 'final_visualizations'
os.makedirs(output_dir, exist_ok=True)

# 1. 相关性热力图
print("生成相关性热力图...")
fig, ax = plt.subplots(figsize=(12, 10))

# 选择要计算相关性的变量
corr_data = merged_data[['llm_sentiment_score_new', 'prev_sentiment', 'daily_return', 'close', 'vol', 'pct_chg']].dropna()

# 计算相关系数矩阵
corr_matrix = corr_data.corr()

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})

plt.title('情感得分与股票市场指标相关性热力图', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 散点图矩阵
print("生成散点图矩阵...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2.1 当日情感得分与当日收益率
ax1 = axes[0, 0]
ax1.scatter(merged_data['llm_sentiment_score_new'], merged_data['daily_return'], alpha=0.6)
ax1.set_xlabel('当日情感得分', fontsize=12)
ax1.set_ylabel('当日收益率 (%)', fontsize=12)
ax1.set_title('当日情感得分与当日收益率散点图', fontsize=14, fontweight='bold')
corr = merged_data['llm_sentiment_score_new'].corr(merged_data['daily_return'])
ax1.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

# 2.2 前一日情感得分与当日收益率
ax2 = axes[0, 1]
ax2.scatter(merged_data['prev_sentiment'], merged_data['daily_return'], alpha=0.6, color='green')
ax2.set_xlabel('前一日情感得分', fontsize=12)
ax2.set_ylabel('当日收益率 (%)', fontsize=12)
ax2.set_title('前一日情感得分与当日收益率散点图', fontsize=14, fontweight='bold')
corr_lag = merged_data['prev_sentiment'].corr(merged_data['daily_return'])
ax2.text(0.05, 0.95, f'相关系数: {corr_lag:.3f}', transform=ax2.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# 2.3 情感得分与成交量
ax3 = axes[1, 0]
ax3.scatter(merged_data['llm_sentiment_score_new'], merged_data['vol'], alpha=0.6, color='orange')
ax3.set_xlabel('当日情感得分', fontsize=12)
ax3.set_ylabel('成交量', fontsize=12)
ax3.set_title('情感得分与成交量散点图', fontsize=14, fontweight='bold')
corr_vol = merged_data['llm_sentiment_score_new'].corr(merged_data['vol'])
ax3.text(0.05, 0.95, f'相关系数: {corr_vol:.3f}', transform=ax3.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# 2.4 情感得分与价格变动
ax4 = axes[1, 1]
ax4.scatter(merged_data['llm_sentiment_score_new'], merged_data['pct_chg'], alpha=0.6, color='purple')
ax4.set_xlabel('当日情感得分', fontsize=12)
ax4.set_ylabel('价格变动 (%)', fontsize=12)
ax4.set_title('情感得分与价格变动散点图', fontsize=14, fontweight='bold')
corr_chg = merged_data['llm_sentiment_score_new'].corr(merged_data['pct_chg'])
ax4.text(0.05, 0.95, f'相关系数: {corr_chg:.3f}', transform=ax4.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 滚动相关性分析
print("生成滚动相关性分析图表...")
fig, ax = plt.subplots(figsize=(14, 8))

# 计算30天滚动相关性
rolling_corr = merged_data['llm_sentiment_score_new'].rolling(window=30).corr(merged_data['daily_return'])
rolling_corr_lag = merged_data['prev_sentiment'].rolling(window=30).corr(merged_data['daily_return'])

# 绘制滚动相关性
ax.plot(range(len(rolling_corr)), rolling_corr, label='当日情感与当日收益率(30天滚动)', linewidth=2)
ax.plot(range(len(rolling_corr_lag)), rolling_corr_lag, label='前日情感与当日收益率(30天滚动)', linewidth=2, color='green')

# 添加零线
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 添加平均线
avg_corr = rolling_corr.mean()
avg_corr_lag = rolling_corr_lag.mean()
ax.axhline(y=avg_corr, color='red', linestyle=':', alpha=0.7, label=f'当日情感平均相关性: {avg_corr:.3f}')
ax.axhline(y=avg_corr_lag, color='green', linestyle=':', alpha=0.7, label=f'前日情感平均相关性: {avg_corr_lag:.3f}')

ax.set_xlabel('时间点', fontsize=12)
ax.set_ylabel('相关系数', fontsize=12)
ax.set_title('情感得分与收益率滚动相关性分析', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/rolling_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 情感分类与收益率箱线图
print("生成情感分类与收益率箱线图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 按情感分类绘制收益率箱线图
sentiment_labels = ['负面', '中性', '正面']
colors = ['#ff9999', '#66b3ff', '#99ff99']

# 创建箱线图数据
boxplot_data = []
boxplot_labels = []
boxplot_colors = []

for label, color in zip(sentiment_labels, colors):
    if label in merged_data['llm_sentiment_label_new'].values:
        data = merged_data[merged_data['llm_sentiment_label_new'] == label]['daily_return']
        boxplot_data.append(data)
        boxplot_labels.append(f'{label}\n(n={len(data)})')
        boxplot_colors.append(color)

# 绘制箱线图
bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)

# 设置颜色
for patch, color in zip(bp['boxes'], boxplot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 添加均值点
means = [data.mean() for data in boxplot_data]
for i, mean in enumerate(means):
    ax.plot(i+1, mean, marker='o', markersize=8, markeredgecolor='black', 
            markerfacecolor='yellow', markeredgewidth=2)

ax.set_xlabel('情感类别', fontsize=12)
ax.set_ylabel('日收益率 (%)', fontsize=12)
ax.set_title('不同情感类别的日收益率分布', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 添加统计信息
stats_text = ""
for i, (label, data) in enumerate(zip(sentiment_labels, boxplot_data)):
    if len(data) > 0:
        mean = data.mean()
        std = data.std()
        stats_text += f"{label}: 均值={mean:.2f}%, 标准差={std:.2f}%\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/sentiment_return_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 相关性显著性检验
print("生成相关性显著性检验图表...")
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 8))

# 计算不同时间窗口的相关性
time_windows = [5, 10, 20, 30, 60]
current_corr = []
lag_corr = []
p_values_current = []
p_values_lag = []

for window in time_windows:
    # 当日情感与当日收益率的相关性
    corr_current, p_current = stats.pearsonr(
        merged_data['llm_sentiment_score_new'].iloc[-window:], 
        merged_data['daily_return'].iloc[-window:]
    )
    current_corr.append(corr_current)
    p_values_current.append(p_current)
    
    # 前一日情感与当日收益率的相关性
    corr_lag, p_lag = stats.pearsonr(
        merged_data['prev_sentiment'].iloc[-window:], 
        merged_data['daily_return'].iloc[-window:]
    )
    lag_corr.append(corr_lag)
    p_values_lag.append(p_lag)

# 创建x轴位置
x_pos = np.arange(len(time_windows))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x_pos - width/2, current_corr, width, label='当日情感与当日收益率', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, lag_corr, width, label='前日情感与当日收益率', alpha=0.7)

# 添加显著性标记
for i, (p1, p2) in enumerate(zip(p_values_current, p_values_lag)):
    if p1 < 0.05:
        ax.text(i-width/2, current_corr[i]+0.01, '*', ha='center', fontsize=12, color='red')
    if p2 < 0.05:
        ax.text(i+width/2, lag_corr[i]+0.01, '*', ha='center', fontsize=12, color='red')

# 添加零线
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

ax.set_xlabel('时间窗口（天）', fontsize=12)
ax.set_ylabel('相关系数', fontsize=12)
ax.set_title('不同时间窗口下的相关性分析', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(time_windows)
ax.legend()
ax.grid(True, alpha=0.3)

# 添加说明
ax.text(0.02, 0.02, '* 表示在5%水平上显著', transform=ax.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_significance.png', dpi=300, bbox_inches='tight')
plt.close()

print("相关性分析图表生成完成！")
print(f"所有图表已保存到 {output_dir} 目录")
