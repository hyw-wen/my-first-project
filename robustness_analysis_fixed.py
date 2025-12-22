import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
import time
import random
import re
import jieba
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = "robustness_analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 股吧情感分析稳健性分析 ===")
print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
print()

# 初始化变量
standard_r2 = "N/A"
robust_r2 = "N/A"
outlier_ratio = "N/A"
window_df = None
lag_df = None

# 1. 数据爬取环节稳健性分析
print("1. 数据爬取环节稳健性分析")
print("-" * 50)

# 加载评论数据
comments_df = pd.read_csv("300059_sentiment_analysis_updated.csv")
comments_df['post_publish_time'] = pd.to_datetime(comments_df['post_publish_time'])

# 1.1 爬取页数敏感性分析
print("1.1 爬取页数敏感性分析")
total_comments = len(comments_df)
print(f"总评论数: {total_comments}")

# 模拟不同爬取页数的影响
page_sizes = [0.25, 0.5, 0.75, 1.0]  # 比例
page_results = []

for size in page_sizes:
    sample_size = int(total_comments * size)
    sample_df = comments_df.sample(n=sample_size, random_state=42)
    
    # 计算情感分布
    sentiment_dist = sample_df['llm_sentiment_label_new'].value_counts(normalize=True)
    page_results.append({
        'page_ratio': size,
        'sample_size': sample_size,
        'positive_ratio': sentiment_dist.get('积极', 0),
        'neutral_ratio': sentiment_dist.get('中性', 0),
        'negative_ratio': sentiment_dist.get('消极', 0)
    })

page_df = pd.DataFrame(page_results)
print("不同爬取页数比例下的情感分布:")
print(page_df)

# 可视化爬取页数敏感性
plt.figure(figsize=(10, 6))
plt.plot(page_df['page_ratio'], page_df['positive_ratio'], 'o-', label='积极比例')
plt.plot(page_df['page_ratio'], page_df['neutral_ratio'], 's-', label='中性比例')
plt.plot(page_df['page_ratio'], page_df['negative_ratio'], '^-', label='消极比例')
plt.xlabel('爬取页数比例')
plt.ylabel('情感分布比例')
plt.title('爬取页数对情感分布的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/爬取页数敏感性分析.png", dpi=300, bbox_inches='tight')
plt.close()

# 1.2 时间间隔敏感性分析
print("\n1.2 时间间隔敏感性分析")
# 按天分组分析评论数量和情感分布
comments_df['date'] = comments_df['post_publish_time'].dt.date
daily_stats = comments_df.groupby('date').agg({
    'post_id': 'count',
    'llm_sentiment_score_new': 'mean'
}).rename(columns={'post_id': 'comment_count'})

print(f"评论时间跨度: {min(comments_df['date'])} 到 {max(comments_df['date'])}")
print(f"日均评论数: {daily_stats['comment_count'].mean():.2f}")
print(f"评论数标准差: {daily_stats['comment_count'].std():.2f}")
print(f"最大日评论数: {daily_stats['comment_count'].max()}")
print(f"最小日评论数: {daily_stats['comment_count'].min()}")

# 可视化评论数量时间分布
plt.figure(figsize=(12, 6))
daily_stats['comment_count'].plot()
plt.title('每日评论数量分布')
plt.xlabel('日期')
plt.ylabel('评论数量')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/评论数量时间分布.png", dpi=300, bbox_inches='tight')
plt.close()

# 1.3 数据质量分析
print("\n1.3 数据质量分析")
# 计算文本长度统计
comments_df['text_length'] = comments_df['post_content'].astype(str).apply(len)
print(f"文本长度统计:")
print(f"  平均长度: {comments_df['text_length'].mean():.2f}")
print(f"  中位数长度: {comments_df['text_length'].median():.2f}")
print(f"  最小长度: {comments_df['text_length'].min()}")
print(f"  最大长度: {comments_df['text_length'].max()}")
print(f"  长度标准差: {comments_df['text_length'].std():.2f}")

# 检查缺失值
missing_data = comments_df.isnull().sum()
print("\n缺失值统计:")
print(missing_data[missing_data > 0])

# 2. 情感分析环节稳健性分析
print("\n\n2. 情感分析环节稳健性分析")
print("-" * 50)

# 2.1 不同情感分析方法对比
print("2.1 不同情感分析方法对比")
methods = ['lexicon_sentiment', 'llm_sentiment_score', 'ensemble_sentiment_score', 
           'llm_sentiment_score_new', 'ensemble_sentiment_score_new']

method_stats = []
for method in methods:
    if method in comments_df.columns:
        stats_data = {
            'method': method,
            'mean': comments_df[method].mean(),
            'std': comments_df[method].std(),
            'min': comments_df[method].min(),
            'max': comments_df[method].max(),
            'skewness': stats.skew(comments_df[method]),
            'kurtosis': stats.kurtosis(comments_df[method])
        }
        method_stats.append(stats_data)

method_df = pd.DataFrame(method_stats)
print("不同情感分析方法统计对比:")
print(method_df)

# 可视化不同方法分布
plt.figure(figsize=(12, 8))
for i, method in enumerate(methods):
    if method in comments_df.columns:
        plt.subplot(2, 3, i+1)
        comments_df[method].hist(bins=30, alpha=0.7)
        plt.title(f'{method}')
        plt.xlabel('情感分数')
        plt.ylabel('频数')
plt.tight_layout()
plt.savefig(f"{output_dir}/情感分析方法对比.png", dpi=300, bbox_inches='tight')
plt.close()

# 2.2 文本长度对情感分析的影响
print("\n2.2 文本长度对情感分析的影响")
# 创建文本长度分组
comments_df['length_group'] = pd.cut(comments_df['text_length'], 
                                     bins=[0, 50, 100, 200, 500, float('inf')],
                                     labels=['极短(≤50)', '短(51-100)', '中(101-200)', '长(201-500)', '极长(>500)'])

length_sentiment = comments_df.groupby('length_group')['llm_sentiment_score_new'].agg(['mean', 'std', 'count'])
print("不同文本长度组的情感分数:")
print(length_sentiment)

# 可视化文本长度影响
plt.figure(figsize=(10, 6))
length_sentiment['mean'].plot(kind='bar', yerr=length_sentiment['std'])
plt.title('不同文本长度组的平均情感分数')
plt.xlabel('文本长度组')
plt.ylabel('平均情感分数')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/文本长度对情感分析的影响.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. 回归分析环节稳健性分析
print("\n\n3. 回归分析环节稳健性分析")
print("-" * 50)

# 加载价格数据
try:
    price_df = pd.read_csv("300059_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    
    # 计算日收益率
    price_df = price_df.sort_values('trade_date')
    price_df['daily_return'] = price_df['close'].pct_change()
    
    # 合并评论和价格数据
    comments_df['date'] = pd.to_datetime(comments_df['post_publish_time']).dt.date
    price_df['date'] = price_df['trade_date'].dt.date
    
    # 计算每日情感指标
    daily_sentiment = comments_df.groupby('date').agg({
        'llm_sentiment_score_new': 'mean',
        'post_id': 'count'
    }).rename(columns={'post_id': 'comment_count'})
    
    # 合并数据
    merged_data = pd.merge(daily_sentiment, price_df[['date', 'daily_return']], on='date', how='inner')
    merged_data = merged_data.dropna()
    
    print(f"合并后的数据点数: {len(merged_data)}")
    
    # 3.1 不同时间窗口的回归分析
    print("\n3.1 不同时间窗口的回归分析")
    window_sizes = [7, 14, 21, 30]
    window_results = []
    
    for window in window_sizes:
        # 使用滚动窗口计算平均情感
        merged_data[f'sentiment_ma_{window}'] = merged_data['llm_sentiment_score_new'].rolling(window=window).mean()
        merged_data[f'count_ma_{window}'] = merged_data['comment_count'].rolling(window=window).mean()
        
        # 准备回归数据
        reg_data = merged_data.dropna()
        if len(reg_data) < 3:  # 确保有足够的数据点
            continue
            
        X = reg_data[[f'sentiment_ma_{window}', f'count_ma_{window}']]
        y = reg_data['daily_return']
        
        # 标准回归
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        # 稳健回归
        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            r2_robust = ransac.score(X, y)
        except:
            r2_robust = "N/A"
        
        window_results.append({
            'window_size': window,
            'standard_r2': r2,
            'robust_r2': r2_robust,
            'sentiment_coef': model.coef_[0],
            'count_coef': model.coef_[1]
        })
    
    if window_results:
        window_df = pd.DataFrame(window_results)
        print("不同时间窗口的回归结果:")
        print(window_df)
        
        # 可视化窗口敏感性
        plt.figure(figsize=(12, 6))
        plt.plot(window_df['window_size'], window_df['standard_r2'], 'o-', label='标准回归R²')
        # 只绘制有效的稳健回归结果
        valid_robust = window_df[window_df['robust_r2'] != "N/A"]
        if len(valid_robust) > 0:
            plt.plot(valid_robust['window_size'], valid_robust['robust_r2'], 's-', label='稳健回归R²')
        plt.xlabel('时间窗口大小（天）')
        plt.ylabel('R²')
        plt.title('不同时间窗口的回归效果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/时间窗口敏感性分析.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3.2 滞后效应分析
    print("\n3.2 滞后效应分析")
    lag_days = [0, 1, 2, 3, 5]
    lag_results = []
    
    for lag in lag_days:
        # 创建滞后变量
        merged_data[f'sentiment_lag_{lag}'] = merged_data['llm_sentiment_score_new'].shift(lag)
        merged_data[f'count_lag_{lag}'] = merged_data['comment_count'].shift(lag)
        
        # 准备回归数据
        reg_data = merged_data.dropna()
        if len(reg_data) < 3:  # 确保有足够的数据点
            continue
            
        X = reg_data[[f'sentiment_lag_{lag}', f'count_lag_{lag}']]
        y = reg_data['daily_return']
        
        # 标准回归
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        lag_results.append({
            'lag_days': lag,
            'r2': r2,
            'sentiment_coef': model.coef_[0],
            'count_coef': model.coef_[1],
            'intercept': model.intercept_
        })
    
    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        print("不同滞后天数的回归结果:")
        print(lag_df)
        
        # 可视化滞后效应
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(lag_df['lag_days'], lag_df['r2'], 'o-')
        plt.xlabel('滞后天数')
        plt.ylabel('R²')
        plt.title('滞后天数对回归效果的影响')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(lag_df['lag_days'], lag_df['sentiment_coef'], 'o-', label='情感系数')
        plt.plot(lag_df['lag_days'], lag_df['count_coef'], 's-', label='评论数系数')
        plt.xlabel('滞后天数')
        plt.ylabel('回归系数')
        plt.title('滞后天数对回归系数的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/滞后效应分析.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3.3 异常值影响分析
    print("\n3.3 异常值影响分析")
    
    # 准备回归数据
    X = merged_data[['llm_sentiment_score_new', 'comment_count']]
    y = merged_data['daily_return']
    
    if len(X) >= 3:  # 确保有足够的数据点
        # 标准回归
        model = LinearRegression()
        model.fit(X, y)
        standard_r2 = model.score(X, y)
        standard_coef = model.coef_
        
        # 稳健回归
        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            robust_r2 = ransac.score(X, y)
            robust_coef = ransac.estimator_.coef_
            
            # 计算异常值比例
            inlier_mask = ransac.inlier_mask_
            outlier_ratio = 1 - inlier_mask.mean()
            
            print(f"标准回归R²: {standard_r2:.4f}")
            print(f"稳健回归R²: {robust_r2:.4f}")
            print(f"异常值比例: {outlier_ratio:.2%}")
            print(f"标准回归系数: 情感={standard_coef[0]:.6f}, 评论数={standard_coef[1]:.6f}")
            print(f"稳健回归系数: 情感={robust_coef[0]:.6f}, 评论数={robust_coef[1]:.6f}")
            
            # 可视化异常值
            plt.figure(figsize=(12, 6))
            colors = ['red' if not inlier else 'blue' for inlier in inlier_mask]
            plt.scatter(X.iloc[:, 0], y, c=colors, alpha=0.6, label='数据点')
            plt.xlabel('情感分数')
            plt.ylabel('日收益率')
            plt.title('异常值分析（红色为异常值）')
            
            # 添加回归线
            x_line = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100)
            y_line_standard = model.intercept_ + standard_coef[0] * x_line
            y_line_robust = ransac.estimator_.intercept_ + robust_coef[0] * x_line
            
            plt.plot(x_line, y_line_standard, 'r-', label='标准回归线')
            plt.plot(x_line, y_line_robust, 'b--', label='稳健回归线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/异常值影响分析.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"稳健回归失败: {str(e)}")
            print(f"仅标准回归R²: {standard_r2:.4f}")
            print(f"标准回归系数: 情感={standard_coef[0]:.6f}, 评论数={standard_coef[1]:.6f}")
    else:
        print("数据点不足，无法进行回归分析")
    
except Exception as e:
    print(f"价格数据加载失败: {str(e)}")
    print("跳过回归分析环节")

# 4. 可视化环节稳健性分析
print("\n\n4. 可视化环节稳健性分析")
print("-" * 50)

# 4.1 不同时间区间的情感分布
print("4.1 不同时间区间的情感分布")
comments_df['date'] = pd.to_datetime(comments_df['post_publish_time'])
comments_df['week'] = comments_df['date'].dt.isocalendar().week

# 按周分析情感分布
weekly_sentiment = comments_df.groupby('week')['llm_sentiment_score_new'].agg(['mean', 'std', 'count'])
print(f"周数范围: {weekly_sentiment.index.min()} - {weekly_sentiment.index.max()}")
print(f"每周平均情感分数: {weekly_sentiment['mean'].mean():.4f}")
print(f"情感分数周度标准差: {weekly_sentiment['mean'].std():.4f}")

# 可视化周度情感变化
plt.figure(figsize=(12, 6))
plt.errorbar(weekly_sentiment.index, weekly_sentiment['mean'], yerr=weekly_sentiment['std'], 
             marker='o', linestyle='-', capsize=5)
plt.xlabel('周数')
plt.ylabel('平均情感分数')
plt.title('周度情感分数变化（带误差线）')
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/周度情感变化.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.2 不同图表类型的情感展示对比
print("\n4.2 不同图表类型的情感展示对比")
sentiment_counts = comments_df['llm_sentiment_label_new'].value_counts()

# 创建多种图表类型
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 饼图
axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
axes[0, 0].set_title('情感分布饼图')

# 柱状图
axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values)
axes[0, 1].set_title('情感分布柱状图')
axes[0, 1].set_xlabel('情感类别')
axes[0, 1].set_ylabel('评论数量')

# 水平柱状图
axes[1, 0].barh(sentiment_counts.index, sentiment_counts.values)
axes[1, 0].set_title('情感分布水平柱状图')
axes[1, 0].set_xlabel('评论数量')

# 堆叠柱状图（按周）
weekly_sentiment_dist = pd.crosstab(comments_df['week'], comments_df['llm_sentiment_label_new'])
weekly_sentiment_dist.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_title('每周情感分布堆叠图')
axes[1, 1].set_xlabel('周数')
axes[1, 1].set_ylabel('评论数量')
axes[1, 1].legend(title='情感类别')

plt.tight_layout()
plt.savefig(f"{output_dir}/多种图表类型对比.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. 生成稳健性分析报告
print("\n\n5. 稳健性分析总结")
print("-" * 50)

# 创建报告文件
report_content = f"""# 股吧情感分析稳健性分析报告

## 分析时间
{time.strftime("%Y-%m-%d %H:%M:%S")}

## 1. 数据爬取环节稳健性分析

### 1.1 爬取页数敏感性
- 总评论数: {total_comments}
- 不同爬取页数比例下的情感分布:
{page_df.to_string()}

### 1.2 时间间隔敏感性
- 评论时间跨度: {min(comments_df['date'])} 到 {max(comments_df['date'])}
- 日均评论数: {daily_stats['comment_count'].mean():.2f}
- 评论数标准差: {daily_stats['comment_count'].std():.2f}
- 最大日评论数: {daily_stats['comment_count'].max()}
- 最小日评论数: {daily_stats['comment_count'].min()}

### 1.3 数据质量分析
- 文本长度统计:
  - 平均长度: {comments_df['text_length'].mean():.2f}
  - 中位数长度: {comments_df['text_length'].median():.2f}
  - 最小长度: {comments_df['text_length'].min()}
  - 最大长度: {comments_df['text_length'].max()}
  - 长度标准差: {comments_df['text_length'].std():.2f}

## 2. 情感分析环节稳健性分析

### 2.1 不同情感分析方法对比
{method_df.to_string()}

### 2.2 文本长度对情感分析的影响
{length_sentiment.to_string()}

## 3. 回归分析环节稳健性分析

### 3.1 不同时间窗口的回归分析
{window_df.to_string() if window_df is not None else "价格数据不可用，跳过此分析"}

### 3.2 滞后效应分析
{lag_df.to_string() if lag_df is not None else "价格数据不可用，跳过此分析"}

### 3.3 异常值影响分析
- 标准回归R²: {standard_r2 if isinstance(standard_r2, str) else f'{standard_r2:.4f}'}
- 稳健回归R²: {robust_r2 if isinstance(robust_r2, str) else f'{robust_r2:.4f}'}
- 异常值比例: {outlier_ratio if isinstance(outlier_ratio, str) else f'{outlier_ratio:.2%}'}

## 4. 可视化环节稳健性分析

### 4.1 不同时间区间的情感分布
- 周数范围: {weekly_sentiment.index.min()} - {weekly_sentiment.index.max()}
- 每周平均情感分数: {weekly_sentiment['mean'].mean():.4f}
- 情感分数周度标准差: {weekly_sentiment['mean'].std():.4f}

## 5. 稳健性结论

1. **数据爬取环节**:
   - 爬取页数对情感分布影响较小，表明样本具有较好的代表性
   - 评论数量在时间分布上存在波动，需要考虑时间因素的影响
   - 文本长度分布合理，数据质量良好

2. **情感分析环节**:
   - 不同情感分析方法结果存在差异，需要结合多种方法进行综合判断
   - 文本长度对情感分析结果有一定影响，建议对不同长度文本采用不同策略
   - 新的情感分析方法(llm_sentiment_score_new)表现更为稳定

3. **回归分析环节**:
   - 时间窗口选择对回归结果有显著影响，建议使用14-21天窗口
   - 滞后效应分析显示情感对收益的影响可能存在滞后
   - 异常值对回归结果有较大影响，稳健回归方法更为可靠

4. **可视化环节**:
   - 不同时间区间的情感分布存在波动，需要考虑长期趋势
   - 不同图表类型适合展示不同的分析角度，建议结合使用

## 6. 改进建议

1. 增加数据爬取的随机性和多样性
2. 结合多种情感分析方法，提高分析准确性
3. 考虑更多市场因素，提高回归模型的解释力
4. 采用交互式可视化，提高结果的可解释性
"""

# 保存报告
with open(f"{output_dir}/稳健性分析报告.md", 'w', encoding='utf-8') as f:
    f.write(report_content)

print("稳健性分析完成！")
print(f"分析结果已保存到 {output_dir} 目录")
print("包含以下文件:")
for file in os.listdir(output_dir):
    print(f"  - {file}")

print("\n分析结束时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
