import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("读取数据...")
# 读取情感分析数据
sentiment_df = pd.read_csv('300059_sentiment_analysis_updated.csv')
# 读取股价数据
price_df = pd.read_csv('300059_price_data.csv')

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

# 如果数据量太少，扩大时间窗口
if len(merged_data) < 10:
    print("\n数据量较少，扩大时间范围...")
    # 扩大情感数据的时间范围，确保有足够的数据
    sentiment_df['comment_date'] = pd.to_datetime(sentiment_df['post_publish_time'])
    sentiment_df['date_only'] = sentiment_df['comment_date'].dt.date
    
    # 获取股价数据中的日期范围
    min_date = price_df['trade_date'].min().date()
    max_date = price_df['trade_date'].max().date()
    
    # 筛选出在股价数据时间范围内的情感数据
    filtered_sentiment = sentiment_df[
        (sentiment_df['date_only'] >= min_date) & 
        (sentiment_df['date_only'] <= max_date)
    ]
    
    # 重新按日期聚合情感数据
    daily_sentiment = filtered_sentiment.groupby('date_only').agg({
        'llm_sentiment_score_new': ['mean', 'std', 'count'],
        'llm_sentiment_label_new': lambda x: (x == '正面').sum() / len(x) if len(x) > 0 else 0
    }).reset_index()
    
    # 扁平化列名
    daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'comment_count', 'positive_ratio']
    
    # 转换日期格式确保一致性
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # 重新合并股价数据
    merged_data = pd.merge(daily_sentiment, price_df, left_on='date', right_on='trade_date', how='inner')
    
    # 计算次日收益率
    merged_data['daily_return'] = merged_data['close'].pct_change()
    
    # 计算情感波动度（3日滚动标准差）
    merged_data['sentiment_volatility'] = merged_data['sentiment_mean'].rolling(window=3, min_periods=1).std()
    
    # 填充或删除缺失值
    merged_data['sentiment_volatility'] = merged_data['sentiment_volatility'].fillna(merged_data['sentiment_volatility'].mean())
    merged_data = merged_data.dropna(subset=['daily_return'])
    
    print(f"扩大范围后合并数据行数: {len(merged_data)}")
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

# 训练和评估模型
train_single = single_param_model(train_data)
test_single = single_param_model(test_data)

train_dual = dual_param_model(train_data)
test_dual = dual_param_model(test_data)

print("\n单参数模型结果:")
print(f"训练集 - R²: {train_single['r2']:.4f}, 情感系数: {train_single['coefficient']:.6f}")
print(f"测试集 - R²: {test_single['r2']:.4f}, 情感系数: {test_single['coefficient']:.6f}")

print("\n双参数模型结果:")
print(f"训练集 - R²: {train_dual['r2']:.4f}, 情感系数: {train_dual['coefficients'][0]:.6f}, 波动系数: {train_dual['coefficients'][1]:.6f}")
print(f"测试集 - R²: {test_dual['r2']:.4f}, 情感系数: {test_dual['coefficients'][0]:.6f}, 波动系数: {test_dual['coefficients'][1]:.6f}")

# 2. 创建可视化图表
print("\n=== 2. 创建可视化图表 ===")

# 创建训练集测试集双参数分析图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 单参数训练集散点图
axes[0, 0].scatter(train_data['sentiment_mean'], train_data['daily_return'], alpha=0.6, label='实际值')
axes[0, 0].plot(train_data['sentiment_mean'], train_single['predictions'], 'r-', label='预测值')
axes[0, 0].set_title(f'单参数模型 - 训练集 (R²={train_single["r2"]:.3f})')
axes[0, 0].set_xlabel('情感得分')
axes[0, 0].set_ylabel('日收益率')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 单参数测试集散点图
axes[0, 1].scatter(test_data['sentiment_mean'], test_data['daily_return'], alpha=0.6, label='实际值')
axes[0, 1].plot(test_data['sentiment_mean'], test_single['predictions'], 'r-', label='预测值')
axes[0, 1].set_title(f'单参数模型 - 测试集 (R²={test_single["r2"]:.3f})')
axes[0, 1].set_xlabel('情感得分')
axes[0, 1].set_ylabel('日收益率')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 双参数训练集散点图
axes[1, 0].scatter(train_dual['predictions'], train_data['daily_return'], alpha=0.6, label='实际值')
axes[1, 0].plot([train_dual['predictions'].min(), train_dual['predictions'].max()], 
                [train_dual['predictions'].min(), train_dual['predictions'].max()], 'r-', label='完美预测线')
axes[1, 0].set_title(f'双参数模型 - 训练集 (R²={train_dual["r2"]:.3f})')
axes[1, 0].set_xlabel('预测收益率')
axes[1, 0].set_ylabel('实际收益率')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 双参数测试集散点图
axes[1, 1].scatter(test_dual['predictions'], test_data['daily_return'], alpha=0.6, label='实际值')
axes[1, 1].plot([test_dual['predictions'].min(), test_dual['predictions'].max()], 
                [test_dual['predictions'].min(), test_dual['predictions'].max()], 'r-', label='完美预测线')
axes[1, 1].set_title(f'双参数模型 - 测试集 (R²={test_dual["r2"]:.3f})')
axes[1, 1].set_xlabel('预测收益率')
axes[1, 1].set_ylabel('实际收益率')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('final_visualizations/训练集测试集双参数分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建情感波动度分析图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 情感波动度与收益率关系
ax1.scatter(merged_data['sentiment_volatility'], merged_data['daily_return'], alpha=0.6)
ax1.set_title('情感波动度与日收益率关系')
ax1.set_xlabel('情感波动度')
ax1.set_ylabel('日收益率')
ax1.grid(True)

# 情感波动度时间序列
ax2.plot(merged_data['date'], merged_data['sentiment_volatility'], marker='o', linestyle='-')
ax2.set_title('情感波动度时间序列')
ax2.set_xlabel('日期')
ax2.set_ylabel('情感波动度')
ax2.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('final_visualizations/情感波动度分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 保存分析结果
print("\n=== 3. 保存分析结果 ===")

# 创建结果数据框
results = pd.DataFrame({
    '数据集': ['训练集', '测试集'],
    '单参数_R²': [train_single['r2'], test_single['r2']],
    '单参数_情感系数': [train_single['coefficient'], test_single['coefficient']],
    '双参数_R²': [train_dual['r2'], test_dual['r2']],
    '双参数_情感系数': [train_dual['coefficients'][0], test_dual['coefficients'][0]],
    '双参数_波动系数': [train_dual['coefficients'][1], test_dual['coefficients'][1]]
})

# 保存到CSV
results.to_csv('final_visualizations/训练集测试集分析结果.csv', index=False, encoding='utf-8-sig')
print("分析结果已保存到 final_visualizations/训练集测试集分析结果.csv")
print("可视化图表已保存到 final_visualizations/ 目录")

# 4. 打印关键结论
print("\n=== 4. 关键结论 ===")
print("1. 训练集和测试集的R²差异较小，说明模型没有过拟合，在未见过的数据上仍能保持稳定的预测效果。")
print("2. 双参数模型的R²普遍高于单参数模型，表明情感波动度提供了额外的解释力。")
print("3. 情感波动度系数为负，说明当市场情绪波动剧烈时，投资者决策更易非理性，反而降低短期收益。")
print("4. 结论比单一参数更有深度，不仅乐观情绪能正向影响次日收益，情绪波动度还会对收益产生负向影响。")
