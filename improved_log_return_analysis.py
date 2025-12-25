#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的对数收益率稳健性分析
解决R方为负值的问题，使用更多数据和更合适的分析方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data():
    """加载并处理数据"""
    print("=== 改进的对数收益率稳健性分析 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载情感分析结果
    comments_df = pd.read_csv("300059_sentiment_analysis_unified.csv")
    print(f"加载情感分析结果: {len(comments_df)} 条评论")
    
    # 加载价格数据
    price_df = pd.read_csv("300059_price_data.csv")
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    print(f"加载价格数据: {len(price_df)} 条记录")
    
    # 计算日收益率和对数收益率
    price_df = price_df.sort_values('trade_date')
    price_df['daily_return'] = price_df['close'].pct_change()
    price_df['log_return'] = np.log(price_df['close'] / price_df['close'].shift(1))
    
    # 处理评论数据，按日期聚合
    comments_df['date'] = pd.to_datetime(comments_df['post_publish_time']).dt.date
    price_df['date'] = price_df['trade_date'].dt.date
    
    # 计算每日情感指标
    daily_sentiment = comments_df.groupby('date').agg({
        'llm_sentiment_score': ['mean', 'std', 'count'],
        'ensemble_sentiment_score': 'mean',
        'lexicon_sentiment': 'mean'
    }).reset_index()
    
    # 扁平化列名
    daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'comment_count', 
                              'ensemble_mean', 'lexicon_mean']
    
    # 合并数据
    merged_data = pd.merge(daily_sentiment, price_df[['date', 'daily_return', 'log_return']], 
                          on='date', how='inner')
    merged_data = merged_data.dropna()
    
    print(f"合并后的数据点数: {len(merged_data)}")
    if len(merged_data) > 0:
        print(f"数据时间范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}")
    
    return merged_data

def improved_regression_analysis(merged_data):
    """改进的回归分析"""
    print("\n1. 改进的回归分析")
    print("-" * 50)
    
    # 准备数据
    X = merged_data['sentiment_mean'].values.reshape(-1, 1)
    y_simple = merged_data['daily_return'].values
    y_log = merged_data['log_return'].values
    
    # 标准线性回归
    model_simple = LinearRegression()
    model_simple.fit(X, y_simple)
    y_simple_pred = model_simple.predict(X)
    r2_simple = r2_score(y_simple, y_simple_pred)
    mse_simple = mean_squared_error(y_simple, y_simple_pred)
    
    model_log = LinearRegression()
    model_log.fit(X, y_log)
    y_log_pred = model_log.predict(X)
    r2_log = r2_score(y_log, y_log_pred)
    mse_log = mean_squared_error(y_log, y_log_pred)
    
    print(f"标准线性回归:")
    print(f"  简单收益率 R²: {r2_simple:.6f}, MSE: {mse_simple:.6f}")
    print(f"  对数收益率 R²: {r2_log:.6f}, MSE: {mse_log:.6f}")
    print(f"  简单收益率系数: {model_simple.coef_[0]:.6f}")
    print(f"  对数收益率系数: {model_log.coef_[0]:.6f}")
    
    # Huber回归（对异常值更稳健）
    huber_simple = HuberRegressor(epsilon=1.35)
    huber_simple.fit(X, y_simple)
    y_simple_huber_pred = huber_simple.predict(X)
    r2_huber_simple = r2_score(y_simple, y_simple_huber_pred)
    
    huber_log = HuberRegressor(epsilon=1.35)
    huber_log.fit(X, y_log)
    y_log_huber_pred = huber_log.predict(X)
    r2_huber_log = r2_score(y_log, y_log_huber_pred)
    
    print(f"\nHuber稳健回归:")
    print(f"  简单收益率 R²: {r2_huber_simple:.6f}")
    print(f"  对数收益率 R²: {r2_huber_log:.6f}")
    print(f"  简单收益率系数: {huber_simple.coef_[0]:.6f}")
    print(f"  对数收益率系数: {huber_log.coef_[0]:.6f}")
    
    # 如果数据点足够，使用RANSAC
    if len(merged_data) >= 10:
        ransac_simple = RANSACRegressor(random_state=42, min_samples=max(5, len(merged_data)//3))
        try:
            ransac_simple.fit(X, y_simple)
            y_simple_ransac_pred = ransac_simple.predict(X)
            r2_ransac_simple = r2_score(y_simple, y_simple_ransac_pred)
            inlier_mask_simple = ransac_simple.inlier_mask_
            print(f"\nRANSAC稳健回归:")
            print(f"  简单收益率 R²: {r2_ransac_simple:.6f}")
            print(f"  简单收益率系数: {ransac_simple.estimator_.coef_[0]:.6f}")
            print(f"  内点数量: {np.sum(inlier_mask_simple)}/{len(merged_data)}")
        except Exception as e:
            print(f"RANSAC简单回归失败: {e}")
            r2_ransac_simple = None
            inlier_mask_simple = None
    else:
        print(f"\n数据点不足({len(merged_data)} < 10)，跳过RANSAC回归")
        r2_ransac_simple = None
        inlier_mask_simple = None
    
    return {
        'simple': {
            'linear': {'r2': r2_simple, 'mse': mse_simple, 'coef': model_simple.coef_[0], 
                      'pred': y_simple_pred},
            'huber': {'r2': r2_huber_simple, 'coef': huber_simple.coef_[0], 
                     'pred': y_simple_huber_pred},
            'ransac': {'r2': r2_ransac_simple, 'inlier_mask': inlier_mask_simple}
        },
        'log': {
            'linear': {'r2': r2_log, 'mse': mse_log, 'coef': model_log.coef_[0], 
                      'pred': y_log_pred},
            'huber': {'r2': r2_huber_log, 'coef': huber_log.coef_[0], 
                     'pred': y_log_huber_pred}
        }
    }

def generate_improved_visualizations(merged_data, regression_results):
    """生成改进的可视化图表"""
    print("\n2. 生成改进的可视化图表")
    print("-" * 50)
    
    # 创建输出目录
    output_dir = "improved_log_return_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建综合图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 子图1: 收益率分布对比
    ax1 = axes[0, 0]
    ax1.hist(merged_data['daily_return'], bins=20, alpha=0.6, color='blue', 
             density=True, label='简单收益率')
    ax1.hist(merged_data['log_return'], bins=20, alpha=0.6, color='red', 
             density=True, label='对数收益率')
    ax1.set_title('收益率分布对比')
    ax1.set_xlabel('收益率')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 简单收益率回归结果
    ax2 = axes[0, 1]
    X = merged_data['sentiment_mean']
    ax2.scatter(X, merged_data['daily_return'], alpha=0.7, color='blue', label='实际值')
    ax2.plot(X, regression_results['simple']['linear']['pred'], 'r-', 
             linewidth=2, label=f'线性回归 (R²={regression_results["simple"]["linear"]["r2"]:.4f})')
    ax2.plot(X, regression_results['simple']['huber']['pred'], 'g--', 
             linewidth=2, label=f'Huber回归 (R²={regression_results["simple"]["huber"]["r2"]:.4f})')
    ax2.set_title('简单收益率 vs 情感得分')
    ax2.set_xlabel('平均情感得分')
    ax2.set_ylabel('简单收益率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 对数收益率回归结果
    ax3 = axes[0, 2]
    ax3.scatter(X, merged_data['log_return'], alpha=0.7, color='red', label='实际值')
    ax3.plot(X, regression_results['log']['linear']['pred'], 'r-', 
             linewidth=2, label=f'线性回归 (R²={regression_results["log"]["linear"]["r2"]:.4f})')
    ax3.plot(X, regression_results['log']['huber']['pred'], 'g--', 
             linewidth=2, label=f'Huber回归 (R²={regression_results["log"]["huber"]["r2"]:.4f})')
    ax3.set_title('对数收益率 vs 情感得分')
    ax3.set_xlabel('平均情感得分')
    ax3.set_ylabel('对数收益率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 收益率时间序列
    ax4 = axes[1, 0]
    ax4.plot(merged_data['date'], merged_data['daily_return'], 'b-', 
             alpha=0.7, label='简单收益率')
    ax4.plot(merged_data['date'], merged_data['log_return'], 'r-', 
             alpha=0.7, label='对数收益率')
    ax4.set_title('收益率时间序列')
    ax4.set_xlabel('日期')
    ax4.set_ylabel('收益率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 子图5: 情感得分时间序列
    ax5 = axes[1, 1]
    ax5.plot(merged_data['date'], merged_data['sentiment_mean'], 'g-', linewidth=2)
    ax5.fill_between(merged_data['date'], 
                     merged_data['sentiment_mean'] - merged_data['sentiment_std'],
                     merged_data['sentiment_mean'] + merged_data['sentiment_std'],
                     alpha=0.3, color='green')
    ax5.set_title('情感得分时间序列')
    ax5.set_xlabel('日期')
    ax5.set_ylabel('情感得分')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # 子图6: 回归系数对比
    ax6 = axes[1, 2]
    methods = ['线性回归', 'Huber回归']
    simple_coefs = [regression_results['simple']['linear']['coef'], 
                   regression_results['simple']['huber']['coef']]
    log_coefs = [regression_results['log']['linear']['coef'], 
                regression_results['log']['huber']['coef']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax6.bar(x - width/2, simple_coefs, width, label='简单收益率', alpha=0.7, color='blue')
    ax6.bar(x + width/2, log_coefs, width, label='对数收益率', alpha=0.7, color='red')
    
    ax6.set_xlabel('回归方法')
    ax6.set_ylabel('回归系数')
    ax6.set_title('回归系数对比')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/改进的稳健性分析.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建回归结果对比表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_data = [
        ['简单收益率', '线性回归', f"{regression_results['simple']['linear']['r2']:.4f}", 
         f"{regression_results['simple']['linear']['coef']:.4f}"],
        ['简单收益率', 'Huber回归', f"{regression_results['simple']['huber']['r2']:.4f}", 
         f"{regression_results['simple']['huber']['coef']:.4f}"],
        ['对数收益率', '线性回归', f"{regression_results['log']['linear']['r2']:.4f}", 
         f"{regression_results['log']['linear']['coef']:.4f}"],
        ['对数收益率', 'Huber回归', f"{regression_results['log']['huber']['r2']:.4f}", 
         f"{regression_results['log']['huber']['coef']:.4f}"]
    ]
    
    # 如果有RANSAC结果，添加到表格中
    if regression_results['simple']['ransac']['r2'] is not None:
        results_data.append(['简单收益率', 'RANSAC回归', 
                            f"{regression_results['simple']['ransac']['r2']:.4f}", 'N/A'])
    
    # 隐藏坐标轴
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=results_data, 
                    colLabels=['收益率类型', '回归方法', 'R²值', '回归系数'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(results_data) + 1):
        for j in range(4):
            cell = table[i, j]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('回归结果对比表', fontsize=16, pad=20)
    plt.savefig(f"{output_dir}/回归结果对比表.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到 {output_dir} 目录")

def generate_improved_report(merged_data, regression_results):
    """生成改进的分析报告"""
    print("\n3. 生成改进的分析报告")
    print("-" * 50)
    
    # 创建输出目录
    output_dir = "improved_log_return_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算统计量
    simple_stats = {
        'mean': merged_data['daily_return'].mean(),
        'std': merged_data['daily_return'].std(),
        'skewness': stats.skew(merged_data['daily_return']),
        'kurtosis': stats.kurtosis(merged_data['daily_return']),
        'min': merged_data['daily_return'].min(),
        'max': merged_data['daily_return'].max()
    }
    
    log_stats = {
        'mean': merged_data['log_return'].mean(),
        'std': merged_data['log_return'].std(),
        'skewness': stats.skew(merged_data['log_return']),
        'kurtosis': stats.kurtosis(merged_data['log_return']),
        'min': merged_data['log_return'].min(),
        'max': merged_data['log_return'].max()
    }
    
    # 正态性检验
    _, p_simple = stats.normaltest(merged_data['daily_return'])
    _, p_log = stats.normaltest(merged_data['log_return'])
    
    # 生成报告内容
    report = f"""# 改进的对数收益率稳健性分析报告

## 1. 数据概况
- 数据点数: {len(merged_data)}
- 时间范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}
- 平均每日评论数: {merged_data['comment_count'].mean():.1f}

## 2. 收益率统计对比

| 统计量 | 简单收益率 | 对数收益率 |
|--------|-----------|-----------|
| 均值 | {simple_stats['mean']:.6f} | {log_stats['mean']:.6f} |
| 标准差 | {simple_stats['std']:.6f} | {log_stats['std']:.6f} |
| 偏度 | {simple_stats['skewness']:.6f} | {log_stats['skewness']:.6f} |
| 峰度 | {simple_stats['kurtosis']:.6f} | {log_stats['kurtosis']:.6f} |
| 最小值 | {simple_stats['min']:.6f} | {log_stats['min']:.6f} |
| 最大值 | {simple_stats['max']:.6f} | {log_stats['max']:.6f} |

## 3. 正态性检验
- 简单收益率正态性检验p值: {p_simple:.6f}
- 对数收益率正态性检验p值: {p_log:.6f}
- 显著性水平α=0.05: {'拒绝正态性假设' if p_simple < 0.05 else '不能拒绝正态性假设'}(简单收益率)
- 显著性水平α=0.05: {'拒绝正态性假设' if p_log < 0.05 else '不能拒绝正态性假设'}(对数收益率)

## 4. 回归分析结果

### 4.1 标准线性回归
- 简单收益率 R²: {regression_results['simple']['linear']['r2']:.6f}
- 对数收益率 R²: {regression_results['log']['linear']['r2']:.6f}
- 简单收益率回归系数: {regression_results['simple']['linear']['coef']:.6f}
- 对数收益率回归系数: {regression_results['log']['linear']['coef']:.6f}

### 4.2 Huber稳健回归
- 简单收益率 R²: {regression_results['simple']['huber']['r2']:.6f}
- 对数收益率 R²: {regression_results['log']['huber']['r2']:.6f}
- 简单收益率回归系数: {regression_results['simple']['huber']['coef']:.6f}
- 对数收益率回归系数: {regression_results['log']['huber']['coef']:.6f}
"""

    # 如果有RANSAC结果，添加到报告中
    if regression_results['simple']['ransac']['r2'] is not None:
        report += f"""
### 4.3 RANSAC稳健回归
- 简单收益率 R²: {regression_results['simple']['ransac']['r2']:.6f}
- 内点数量: {np.sum(regression_results['simple']['ransac']['inlier_mask'])}/{len(merged_data)}
"""

    report += f"""
## 5. 分析结论

### 5.1 收益率分布特性
1. 对数收益率的偏度({log_stats['skewness']:.4f}){'小于' if log_stats['skewness'] < simple_stats['skewness'] else '大于'}简单收益率的偏度({simple_stats['skewness']:.4f})
2. 对数收益率的峰度({log_stats['kurtosis']:.4f}){'小于' if log_stats['kurtosis'] < simple_stats['kurtosis'] else '大于'}简单收益率的峰度({simple_stats['kurtosis']:.4f})
3. {'对数收益率更接近正态分布' if abs(log_stats['skewness']) < abs(simple_stats['skewness']) else '简单收益率更接近正态分布'}

### 5.2 回归分析结论
1. 标准线性回归中，{'对数收益率' if regression_results['log']['linear']['r2'] > regression_results['simple']['linear']['r2'] else '简单收益率'}的R²值更高
2. Huber稳健回归中，{'对数收益率' if regression_results['log']['huber']['r2'] > regression_results['simple']['huber']['r2'] else '简单收益率'}的R²值更高
3. {'对数收益率' if abs(regression_results['log']['linear']['coef']) > abs(regression_results['simple']['linear']['coef']) else '简单收益率'}对情感得分更敏感

### 5.3 稳健性评估
1. {'对数收益率模型更稳健' if (regression_results['log']['linear']['r2'] > regression_results['simple']['linear']['r2'] and 
                               regression_results['log']['huber']['r2'] > regression_results['simple']['huber']['r2']) else '简单收益率模型更稳健'}
2. 推荐使用{'对数收益率' if regression_results['log']['linear']['r2'] > regression_results['simple']['linear']['r2'] else '简单收益率'}进行情感分析建模

## 6. 建议
1. 使用{'对数收益率' if regression_results['log']['linear']['r2'] > regression_results['simple']['linear']['r2'] else '简单收益率'}作为主要分析指标
2. 考虑使用Huber回归作为主要回归方法，因为它对异常值更稳健
3. 建议收集更多数据点以提高模型的可靠性
4. 可以考虑添加其他解释变量（如市场指数、交易量等）来提高模型解释力

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告
    with open(f"{output_dir}/改进的对数收益率分析报告.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已保存到 {output_dir}/改进的对数收益率分析报告.md")

def main():
    """主函数"""
    try:
        # 加载和处理数据
        merged_data = load_and_process_data()
        
        if len(merged_data) < 3:
            print("错误: 数据点太少，无法进行有意义的分析")
            return
        
        # 改进的回归分析
        regression_results = improved_regression_analysis(merged_data)
        
        # 生成改进的可视化
        generate_improved_visualizations(merged_data, regression_results)
        
        # 生成改进的报告
        generate_improved_report(merged_data, regression_results)
        
        print(f"\n改进的对数收益率稳健性分析完成！")
        print(f"分析结果已保存到 improved_log_return_analysis 目录")
        print(f"包含以下文件:")
        print(f"  - 改进的对数收益率分析报告.md")
        print(f"  - 改进的稳健性分析.png")
        print(f"  - 回归结果对比表.png")
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
