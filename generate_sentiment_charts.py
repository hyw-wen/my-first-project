import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图表保存目录
import os
if not os.path.exists('final_visualizations'):
    os.makedirs('final_visualizations')

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

# 1. 情感分布图（饼图和柱状图）
print("生成情感分布图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 使用最新的情感分析结果
sentiment_counts = df['llm_sentiment_label_new'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99']

# 饼图
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12})
ax1.set_title('股吧评论情感分布（饼图）', fontsize=14, fontweight='bold')

# 柱状图
bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
ax2.set_title('股吧评论情感分布（柱状图）', fontsize=14, fontweight='bold')
ax2.set_xlabel('情感类别', fontsize=12)
ax2.set_ylabel('评论数量', fontsize=12)

# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('final_visualizations/情感分布图.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 情感得分分布直方图
print("生成情感得分分布图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 情感得分分布直方图
ax1.hist(df['llm_sentiment_score_new'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('情感得分分布直方图', fontsize=14, fontweight='bold')
ax1.set_xlabel('情感得分', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='中性线')
ax1.legend()

# 情感得分密度图
sns.kdeplot(df['llm_sentiment_score_new'], ax=ax2, shade=True, color='purple')
ax2.set_title('情感得分密度图', fontsize=14, fontweight='bold')
ax2.set_xlabel('情感得分', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='中性线')
ax2.legend()

plt.tight_layout()
plt.savefig('final_visualizations/情感得分分布图.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 评论数量时间分布图
print("生成评论数量时间分布图...")
# 按日期聚合评论数量
df['date'] = df['post_publish_time'].dt.date
daily_comments = df.groupby('date').size().reset_index(name='comment_count')
daily_comments['date'] = pd.to_datetime(daily_comments['date'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 每日评论数量柱状图
ax1.bar(daily_comments['date'], daily_comments['comment_count'], color='lightcoral', alpha=0.7)
ax1.set_title('每日评论数量分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('评论数量', fontsize=12)
ax1.grid(True, alpha=0.3)

# 添加移动平均线
window_size = 3
daily_comments['moving_avg'] = daily_comments['comment_count'].rolling(window=window_size).mean()
ax1.plot(daily_comments['date'], daily_comments['moving_avg'], color='blue', linewidth=2, label=f'{window_size}日移动平均')
ax1.legend()

# 按小时聚合评论数量
df['hour'] = df['post_publish_time'].dt.hour
hourly_comments = df.groupby('hour').size().reset_index(name='comment_count')

# 每小时评论数量柱状图
ax2.bar(hourly_comments['hour'], hourly_comments['comment_count'], color='lightgreen', alpha=0.7)
ax2.set_title('每小时评论数量分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('小时', fontsize=12)
ax2.set_ylabel('评论数量', fontsize=12)
ax2.set_xticks(range(0, 24))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/评论数量时间分布图.png', dpi=300, bbox_inches='tight')
plt.close()

print("情感分布图生成完成！")
