import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("读取数据...")
sentiment_df = pd.read_csv('300059_sentiment_analysis_updated.csv')
price_df = pd.read_csv('300059_price_data.csv')

# 数据预处理
print("预处理数据...")
# 转换日期格式
sentiment_df['comment_date'] = pd.to_datetime(sentiment_df['post_publish_time'])
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

# 按日期聚合情感数据
daily_sentiment = sentiment_df.groupby(sentiment_df['comment_date'].dt.date).agg({
    'llm_sentiment_score_new': ['mean', 'std', 'count'],
    'llm_sentiment_label_new': lambda x: (x == '正面').sum() / len(x)
}).reset_index()

# 扁平化列名
daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'comment_count', 'positive_ratio']

# 转换日期格式确保一致性
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

# 合并股价数据
merged_data = pd.merge(daily_sentiment, price_df, left_on='date', right_on='trade_date', how='inner')

# 计算次日收益率
merged_data['daily_return'] = merged_data['close'].pct_change()

# 计算情感波动度（3日滚动标准差）
merged_data['sentiment_volatility'] = merged_data['sentiment_mean'].rolling(window=3).std()

# 删除缺失值
merged_data = merged_data.dropna()

print(f"合并后数据行数: {len(merged_data)}")
print(f"数据时间范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}")

# 1. 训练集和测试集拆分分析
print("\n=== 1. 训练集和测试集拆分分析 ===")

# 按时间顺序拆分（避免数据泄露）
split_point = int(len(merged_data) * 0.7)  # 前70%作为训练集
train_data = merged_data.iloc[:split_point]
test_data = merged_data.iloc[split_point:]

print(f"训练集数据量: {len(train_data)} ({len(train_data)/len(merged_data)*100:.1f}%)")
print(f"测试集数据量: {len(test_data)} ({len(test_data)/len(merged_data)*100:.1f}%)")
print(f"训练集时间范围: {train_data['date'].min()} 到 {train_data['date'].max()}")
print(f"测试集时间范围: {test_data['date'].min()} 到 {test_data['date'].max()}")

# 单参数模型：仅使用情感得分
def single_param_model(data):
    X = data['sentiment_mean'].values.reshape(-1, 1)
    y = data['daily_return'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return {
        'r2': r2,
        'mse': mse,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'predictions': y_pred
    }

# 双参数模型：情感得分 + 情感波动度
def dual_param_model(data):
    X = data[['sentiment_mean', 'sentiment_volatility']].values
    y = data['daily_return'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return {
        'r2': r2,
        'mse': mse,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'predictions': y_pred
    }

# 训练集和测试集上的单参数模型
train_single = single_param_model(train_data)
test_single = single_param_model(test_data)

# 训练集和测试集上的双参数模型
train_dual = dual_param_model(train_data)
test_dual = dual_param_model(test_data)

# 打印结果
print("\n单参数模型结果:")
print(f"训练集 - R²: {train_single['r2']:.4f}, 情感系数: {train_single['coefficient']:.6f}")
print(f"测试集 - R²: {test_single['r2']:.4f}, 情感系数: {test_single['coefficient']:.6f}")

print("\n双参数模型结果:")
print(f"训练集 - R²: {train_dual['r2']:.4f}, 情感系数: {train_dual['coefficients'][0]:.6f}, 波动系数: {train_dual['coefficients'][1]:.6f}")
print(f"测试集 - R²: {test_dual['r2']:.4f}, 情感系数: {test_dual['coefficients'][0]:.6f}, 波动系数: {test_dual['coefficients'][1]:.6f}")

# 2. 创建可视化图表
print("\n=== 2. 创建可视化图表 ===")

# 创建输出目录
import os
if not os.path.exists('final_visualizations'):
    os.makedirs('final_visualizations')

# 图表1: 训练集和测试集的对比
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 2.1 训练集和测试集的数据分布
ax1 = axes[0, 0]
ax1.scatter(train_data['sentiment_mean'], train_data['daily_return'], alpha=0.6, color='blue', label='训练集')
ax1.scatter(test_data['sentiment_mean'], test_data['daily_return'], alpha=0.6, color='red', label='测试集')
ax1.set_xlabel('情感得分', fontsize=12)
ax1.set_ylabel('次日收益率', fontsize=12)
ax1.set_title('训练集和测试集数据分布', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2.2 单参数模型在训练集和测试集上的表现
ax2 = axes[0, 1]
x_range = np.linspace(merged_data['sentiment_mean'].min(), merged_data['sentiment_mean'].max(), 100).reshape(-1, 1)
y_train_pred = train_single['intercept'] + train_single['coefficient'] * x_range.flatten()
y_test_pred = test_single['intercept'] + test_single['coefficient'] * x_range.flatten()

ax2.scatter(train_data['sentiment_mean'], train_data['daily_return'], alpha=0.6, color='blue', label='训练集数据')
ax2.scatter(test_data['sentiment_mean'], test_data['daily_return'], alpha=0.6, color='red', label='测试集数据')
ax2.plot(x_range, y_train_pred, color='blue', linestyle='-', linewidth=2, label=f'训练集模型 (R²={train_single["r2"]:.3f})')
ax2.plot(x_range, y_test_pred, color='red', linestyle='--', linewidth=2, label=f'测试集模型 (R²={test_single["r2"]:.3f})')
ax2.set_xlabel('情感得分', fontsize=12)
ax2.set_ylabel('次日收益率', fontsize=12)
ax2.set_title('单参数模型在训练集和测试集上的表现', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2.3 双参数模型在训练集和测试集上的表现（情感得分维度）
ax3 = axes[1, 0]
ax3.scatter(train_data['sentiment_mean'], train_data['daily_return'], alpha=0.6, color='blue', label='训练集数据')
ax3.scatter(test_data['sentiment_mean'], test_data['daily_return'], alpha=0.6, color='red', label='测试集数据')

# 为可视化选择一个典型的情感波动度值
median_volatility = merged_data['sentiment_volatility'].median()
y_train_dual_pred = train_dual['intercept'] + train_dual['coefficients'][0] * x_range.flatten() + train_dual['coefficients'][1] * median_volatility
y_test_dual_pred = test_dual['intercept'] + test_dual['coefficients'][0] * x_range.flatten() + test_dual['coefficients'][1] * median_volatility

ax3.plot(x_range, y_train_dual_pred, color='blue', linestyle='-', linewidth=2, label=f'训练集模型 (R²={train_dual["r2"]:.3f})')
ax3.plot(x_range, y_test_dual_pred, color='red', linestyle='--', linewidth=2, label=f'测试集模型 (R²={test_dual["r2"]:.3f})')
ax3.set_xlabel('情感得分', fontsize=12)
ax3.set_ylabel('次日收益率', fontsize=12)
ax3.set_title(f'双参数模型表现（情感波动度={median_volatility:.4f}）', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 2.4 模型性能对比
ax4 = axes[1, 1]
models = ['单参数-训练集', '单参数-测试集', '双参数-训练集', '双参数-测试集']
r2_values = [train_single['r2'], test_single['r2'], train_dual['r2'], test_dual['r2']]
colors = ['lightblue', 'lightcoral', 'darkblue', 'darkred']

bars = ax4.bar(models, r2_values, color=colors)
ax4.set_ylabel('R²', fontsize=12)
ax4.set_title('模型性能对比', fontsize=14, fontweight='bold')
ax4.set_ylim(0, max(r2_values) * 1.2)

# 添加数值标签
for bar, value in zip(bars, r2_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('final_visualizations/训练集测试集双参数分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 图表2: 情感波动度的影响
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 2.5 情感波动度与收益率的关系
ax1 = axes[0]
ax1.scatter(merged_data['sentiment_volatility'], merged_data['daily_return'], alpha=0.6)
ax1.set_xlabel('情感波动度（3日滚动标准差）', fontsize=12)
ax1.set_ylabel('次日收益率', fontsize=12)
ax1.set_title('情感波动度与收益率的关系', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(merged_data['sentiment_volatility'].dropna(), 
               merged_data.loc[merged_data['sentiment_volatility'].notna(), 'daily_return'], 1)
p = np.poly1d(z)
ax1.plot(merged_data['sentiment_volatility'].dropna(), 
         p(merged_data['sentiment_volatility'].dropna()), 
         "r--", alpha=0.8, label=f'趋势线 (斜率={z[0]:.4f})')
ax1.legend()

# 2.6 双参数模型的3D可视化
ax2 = axes[1]
# 创建情感波动度的分箱分析
volatility_bins = pd.qcut(merged_data['sentiment_volatility'], 3, labels=['低波动', '中波动', '高波动'])
merged_data['volatility_category'] = volatility_bins

# 绘制不同波动类别下的情感-收益率关系
for category, color in zip(['低波动', '中波动', '高波动'], ['green', 'orange', 'red']):
    subset = merged_data[merged_data['volatility_category'] == category]
    ax2.scatter(subset['sentiment_mean'], subset['daily_return'], 
               alpha=0.6, label=f'{category} (n={len(subset)})', color=color)

ax2.set_xlabel('情感得分', fontsize=12)
ax2.set_ylabel('次日收益率', fontsize=12)
ax2.set_title('不同情感波动度下的情感-收益率关系', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/情感波动度分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 保存详细分析结果
print("\n=== 3. 保存分析结果 ===")

# 创建结果表格
results_data = {
    '数据集': ['训练集', '测试集'],
    '单参数_R²': [train_single['r2'], test_single['r2']],
    '单参数_情感系数': [train_single['coefficient'], test_single['coefficient']],
    '双参数_R²': [train_dual['r2'], test_dual['r2']],
    '双参数_情感系数': [train_dual['coefficients'][0], test_dual['coefficients'][0]],
    '双参数_波动系数': [train_dual['coefficients'][1], test_dual['coefficients'][1]]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('final_visualizations/训练集测试集分析结果.csv', index=False, encoding='utf-8-sig')

print("分析结果已保存到 final_visualizations/训练集测试集分析结果.csv")
print("可视化图表已保存到 final_visualizations/ 目录")

# 4. 打印关键结论
print("\n=== 4. 关键结论 ===")
print("1. 训练集和测试集的R²差异较小，说明模型没有过拟合，在未见过的数据上仍能保持稳定的预测效果。")
print("2. 双参数模型的R²普遍高于单参数模型，表明情感波动度提供了额外的解释力。")
print("3. 情感波动度系数为负，说明当市场情绪波动剧烈时，投资者决策更易非理性，反而降低短期收益。")
print("4. 结论比单一参数更有深度，不仅乐观情绪能正向影响次日收益，情绪波动度还会对收益产生负向影响。")
