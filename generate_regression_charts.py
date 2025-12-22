import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

# 创建滞后数据
for lag in range(1, 6):
    merged_data[f'sentiment_lag_{lag}'] = merged_data['llm_sentiment_score_new'].shift(lag)

# 1. 多元线性回归分析
print("生成多元线性回归分析图表...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 当日情感与当日收益率的回归分析
ax1 = axes[0, 0]
X = merged_data[['llm_sentiment_score_new']].dropna()
y = merged_data.loc[X.index, 'daily_return']

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# 计算置信区间
n = len(X)
t = stats.t.ppf(0.975, n-2)  # 95%置信区间
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(x_range)
residuals = y - y_pred
mse_res = np.sum(residuals**2) / (n-2)
x_mean = X.mean().values[0]  # 获取标量值
se_pred = np.sqrt(mse_res * (1/n + (x_range.flatten() - x_mean)**2 / np.sum((X.values.flatten() - x_mean)**2)))
ci_low = y_range_pred.flatten() - t * se_pred
ci_high = y_range_pred.flatten() + t * se_pred

# 绘制散点图和回归线
ax1.scatter(X, y, alpha=0.6, color='blue')
ax1.plot(x_range, y_range_pred, color='red', linewidth=2, label=f'回归线 (R²={r2:.3f})')
ax1.fill_between(x_range.flatten(), ci_low.flatten(), ci_high.flatten(), color='red', alpha=0.2, label='95%置信区间')
ax1.set_xlabel('情感得分', fontsize=12)
ax1.set_ylabel('日收益率', fontsize=12)
ax1.set_title('当日情感得分与日收益率回归分析', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', transform=ax1.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1.2 前一日情感与当日收益率的回归分析
ax2 = axes[0, 1]
X_lag = merged_data[['sentiment_lag_1']].dropna()
y_lag = merged_data.loc[X_lag.index, 'daily_return']

# 拟合线性回归模型
model_lag = LinearRegression()
model_lag.fit(X_lag, y_lag)
y_pred_lag = model_lag.predict(X_lag)
r2_lag = r2_score(y_lag, y_pred_lag)
mse_lag = mean_squared_error(y_lag, y_pred_lag)
mae_lag = mean_absolute_error(y_lag, y_pred_lag)

# 计算置信区间
n_lag = len(X_lag)
t_lag = stats.t.ppf(0.975, n_lag-2)  # 95%置信区间
x_range_lag = np.linspace(X_lag.min(), X_lag.max(), 100).reshape(-1, 1)
y_range_pred_lag = model_lag.predict(x_range_lag)
residuals_lag = y_lag - y_pred_lag
mse_res_lag = np.sum(residuals_lag**2) / (n_lag-2)
x_mean_lag = X_lag.mean().values[0]  # 获取标量值
se_pred_lag = np.sqrt(mse_res_lag * (1/n_lag + (x_range_lag.flatten() - x_mean_lag)**2 / np.sum((X_lag.values.flatten() - x_mean_lag)**2)))
ci_low_lag = y_range_pred_lag.flatten() - t_lag * se_pred_lag
ci_high_lag = y_range_pred_lag.flatten() + t_lag * se_pred_lag

# 绘制散点图和回归线
ax2.scatter(X_lag, y_lag, alpha=0.6, color='green')
ax2.plot(x_range_lag, y_range_pred_lag, color='red', linewidth=2, label=f'回归线 (R²={r2_lag:.3f})')
ax2.fill_between(x_range_lag.flatten(), ci_low_lag.flatten(), ci_high_lag.flatten(), color='red', alpha=0.2, label='95%置信区间')
ax2.set_xlabel('前一日情感得分', fontsize=12)
ax2.set_ylabel('当日收益率', fontsize=12)
ax2.set_title('前一日情感得分与当日收益率回归分析', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'MSE: {mse_lag:.4f}\nMAE: {mae_lag:.4f}', transform=ax2.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1.3 多元回归分析（当日情感和前一日情感）
ax3 = axes[1, 0]
X_multi = merged_data[['llm_sentiment_score_new', 'sentiment_lag_1']].dropna()
y_multi = merged_data.loc[X_multi.index, 'daily_return']

# 拟合多元线性回归模型
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)
y_pred_multi = model_multi.predict(X_multi)
r2_multi = r2_score(y_multi, y_pred_multi)
mse_multi = mean_squared_error(y_multi, y_pred_multi)
mae_multi = mean_absolute_error(y_multi, y_pred_multi)

# 绘制实际值与预测值对比
ax3.scatter(y_multi, y_pred_multi, alpha=0.6, color='purple')
ax3.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', lw=2)
ax3.set_xlabel('实际收益率', fontsize=12)
ax3.set_ylabel('预测收益率', fontsize=12)
ax3.set_title('多元回归预测效果', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, f'R²: {r2_multi:.3f}\nMSE: {mse_multi:.4f}\nMAE: {mae_multi:.4f}', transform=ax3.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1.4 回归系数比较
ax4 = axes[1, 1]
models = ['单日情感', '前一日情感', '多元回归']
r2_values = [r2, r2_lag, r2_multi]
mse_values = [mse, mse_lag, mse_multi]

# 创建双轴图表
ax4_twin = ax4.twinx()

# 绘制R²柱状图
bars1 = ax4.bar(models, r2_values, alpha=0.7, color='blue', label='R²')
ax4.set_ylabel('R²', fontsize=12, color='blue')
ax4.tick_params(axis='y', labelcolor='blue')
ax4.set_ylim(0, max(r2_values) * 1.2)

# 绘制MSE线图
line1, = ax4_twin.plot(models, mse_values, 'ro-', linewidth=2, markersize=8, label='MSE')
ax4_twin.set_ylabel('MSE', fontsize=12, color='red')
ax4_twin.tick_params(axis='y', labelcolor='red')

# 添加图例
lines = [bars1, line1]
labels = ['R²', 'MSE']
ax4.legend(lines, labels, loc='upper left')
ax4.set_title('回归模型性能比较', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar, r2_val in zip(bars1, r2_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{r2_val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('final_visualizations/回归分析图表.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 残差分析
print("生成残差分析图表...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2.1 单日情感回归的残差图
ax1 = axes[0, 0]
residuals = y - y_pred
ax1.scatter(y_pred, residuals, alpha=0.6, color='blue')
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_xlabel('预测值', fontsize=12)
ax1.set_ylabel('残差', fontsize=12)
ax1.set_title('单日情感回归残差图', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2.2 前一日情感回归的残差图
ax2 = axes[0, 1]
residuals_lag = y_lag - y_pred_lag
ax2.scatter(y_pred_lag, residuals_lag, alpha=0.6, color='green')
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_xlabel('预测值', fontsize=12)
ax2.set_ylabel('残差', fontsize=12)
ax2.set_title('前一日情感回归残差图', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 2.3 残差正态性检验（Q-Q图）
ax3 = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('单日情感回归残差Q-Q图', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 2.4 残差直方图
ax4 = axes[1, 1]
ax4.hist(residuals, bins=20, alpha=0.7, color='blue', density=True)
# 拟合正态分布
mu, std = stats.norm.fit(residuals)
xmin, xmax = ax4.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
ax4.plot(x, p, 'k', linewidth=2, label=f'正态分布拟合 (μ={mu:.4f}, σ={std:.4f})')
ax4.set_xlabel('残差', fontsize=12)
ax4.set_ylabel('密度', fontsize=12)
ax4.set_title('残差分布直方图', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/残差分析图表.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 回归诊断
print("生成回归诊断图表...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 杠杆值分析
ax1 = axes[0, 0]
# 计算杠杆值
X_with_intercept = np.column_stack([np.ones(len(X)), X])
H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
leverage = np.diag(H)
ax1.scatter(range(len(leverage)), leverage, alpha=0.6, color='blue')
ax1.axhline(y=2*len(X.columns)/len(X), color='red', linestyle='--', label='高杠杆阈值')
ax1.set_xlabel('观测值索引', fontsize=12)
ax1.set_ylabel('杠杆值', fontsize=12)
ax1.set_title('杠杆值分析', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 Cook距离分析
ax2 = axes[0, 1]
# 计算Cook距离
residuals = y - y_pred
mse = np.sum(residuals**2) / (len(X) - len(X.columns) - 1)
cooks_d = residuals**2 / (mse * len(X.columns)) * leverage / (1 - leverage)**2
ax2.scatter(range(len(cooks_d)), cooks_d, alpha=0.6, color='green')
ax2.axhline(y=4/len(X), color='red', linestyle='--', label='高影响点阈值')
ax2.set_xlabel('观测值索引', fontsize=12)
ax2.set_ylabel('Cook距离', fontsize=12)
ax2.set_title('Cook距离分析', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 标准化残差分析
ax3 = axes[1, 0]
standardized_residuals = residuals / np.std(residuals)
ax3.scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6, color='purple')
ax3.axhline(y=2, color='red', linestyle='--', label='±2标准差')
ax3.axhline(y=-2, color='red', linestyle='--')
ax3.set_xlabel('观测值索引', fontsize=12)
ax3.set_ylabel('标准化残差', fontsize=12)
ax3.set_title('标准化残差分析', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4 残差与自变量关系
ax4 = axes[1, 1]
ax4.scatter(X.values.flatten(), residuals, alpha=0.6, color='orange')
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_xlabel('情感得分', fontsize=12)
ax4.set_ylabel('残差', fontsize=12)
ax4.set_title('残差与自变量关系', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_visualizations/回归诊断图表.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 模型稳定性分析
print("生成模型稳定性分析图表...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.1 不同时间窗口的回归系数稳定性
ax1 = axes[0, 0]
window_sizes = [30, 45, 60, 90]  # 不同的时间窗口
coefficients = []
p_values = []

for window in window_sizes:
    if len(merged_data) >= window:
        # 滑动窗口回归
        coeffs = []
        p_vals = []
        for i in range(len(merged_data) - window + 1):
            window_data = merged_data.iloc[i:i+window]
            X_window = window_data[['llm_sentiment_score_new']].dropna()
            y_window = window_data.loc[X_window.index, 'daily_return']
            
            if len(X_window) > 1:
                model = LinearRegression()
                model.fit(X_window, y_window)
                coeffs.append(model.coef_[0])
                
                # 计算p值
                n = len(X_window)
                mse = np.sum((y_window - model.predict(X_window))**2) / (n - 2)
                var_coef = mse / np.sum((X_window.values.flatten() - X_window.values.flatten().mean())**2)
                t_stat = model.coef_[0] / np.sqrt(var_coef)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                p_vals.append(p_val)
        
        coefficients.append(coeffs)
        p_values.append(p_vals)

# 绘制系数稳定性图
for i, window in enumerate(window_sizes):
    if i < len(coefficients) and len(coefficients[i]) > 0:
        ax1.plot(range(len(coefficients[i])), coefficients[i], label=f'{window}天窗口', alpha=0.7)

ax1.set_xlabel('时间窗口索引', fontsize=12)
ax1.set_ylabel('回归系数', fontsize=12)
ax1.set_title('不同时间窗口的回归系数稳定性', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4.2 不同滞后期的回归系数比较
ax2 = axes[0, 1]
lag_coeffs = []
lag_r2 = []
lag_p_values = []

for lag in range(1, 6):
    X_lag = merged_data[[f'sentiment_lag_{lag}']].dropna()
    y_lag = merged_data.loc[X_lag.index, 'daily_return']
    
    if len(X_lag) > 1:
        model = LinearRegression()
        model.fit(X_lag, y_lag)
        lag_coeffs.append(model.coef_[0])
        lag_r2.append(r2_score(y_lag, model.predict(X_lag)))
        
        # 计算p值
        n = len(X_lag)
        mse = np.sum((y_lag - model.predict(X_lag))**2) / (n - 2)
        var_coef = mse / np.sum((X_lag.values.flatten() - X_lag.values.flatten().mean())**2)
        t_stat = model.coef_[0] / np.sqrt(var_coef)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        lag_p_values.append(p_val)

# 绘制滞后系数图
lags = list(range(1, 6))
bars = ax2.bar(lags, lag_coeffs, alpha=0.7, color='skyblue')
ax2.set_xlabel('滞后期（天）', fontsize=12)
ax2.set_ylabel('回归系数', fontsize=12)
ax2.set_title('不同滞后期回归系数比较', fontsize=14, fontweight='bold')
ax2.set_xticks(lags)
ax2.grid(True, alpha=0.3)

# 在柱状图上添加显著性标记
for bar, coeff, p_val in zip(bars, lag_coeffs, lag_p_values):
    height = bar.get_height()
    sig_level = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
             f'{coeff:.3f}{sig_level}', ha='center', va='bottom' if height >= 0 else 'top')

# 4.3 不同数据子集的回归系数稳定性
ax3 = axes[1, 0]
# 按月份分组
merged_data['month'] = merged_data['date'].dt.month
month_coeffs = []
month_r2 = []

for month in sorted(merged_data['month'].unique()):
    month_data = merged_data[merged_data['month'] == month]
    X_month = month_data[['llm_sentiment_score_new']].dropna()
    y_month = month_data.loc[X_month.index, 'daily_return']
    
    if len(X_month) > 1:
        model = LinearRegression()
        model.fit(X_month, y_month)
        month_coeffs.append(model.coef_[0])
        month_r2.append(r2_score(y_month, model.predict(X_month)))

# 绘制月份系数图
months = sorted(merged_data['month'].unique())
bars = ax3.bar(months, month_coeffs, alpha=0.7, color='lightgreen')
ax3.set_xlabel('月份', fontsize=12)
ax3.set_ylabel('回归系数', fontsize=12)
ax3.set_title('不同月份回归系数比较', fontsize=14, fontweight='bold')
ax3.set_xticks(months)
ax3.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar, coeff in zip(bars, month_coeffs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
             f'{coeff:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

# 4.4 回归系数置信区间比较
ax4 = axes[1, 1]
# 计算不同模型的系数置信区间
models_info = [
    ('单日情感', X, y),
    ('前一日情感', X_lag, y_lag),
    ('多元回归', X_multi, y_multi)
]

model_names = []
coef_means = []
coef_lower = []
coef_upper = []

for name, X_data, y_data in models_info:
    if len(X_data) > 1:
        model = LinearRegression()
        model.fit(X_data, y_data)
        
        # 计算置信区间
        n = len(X_data)
        mse = np.sum((y_data - model.predict(X_data))**2) / (n - len(X_data.columns))
        
        if len(X_data.columns) == 1:
            # 单变量回归
            var_coef = mse / np.sum((X_data.values.flatten() - X_data.values.flatten().mean())**2)
            se_coef = np.sqrt(var_coef)
            t_val = stats.t.ppf(0.975, n-2)
            
            model_names.append(name)
            coef_means.append(model.coef_[0])
            coef_lower.append(model.coef_[0] - t_val * se_coef)
            coef_upper.append(model.coef_[0] + t_val * se_coef)
        else:
            # 多变量回归，只显示第一个变量的置信区间
            var_coef = mse / np.sum((X_data.iloc[:, 0].values - X_data.iloc[:, 0].values.mean())**2)
            se_coef = np.sqrt(var_coef)
            t_val = stats.t.ppf(0.975, n-len(X_data.columns))
            
            model_names.append(f"{name}(情感)")
            coef_means.append(model.coef_[0])
            coef_lower.append(model.coef_[0] - t_val * se_coef)
            coef_upper.append(model.coef_[0] + t_val * se_coef)

# 绘制置信区间图
x_pos = np.arange(len(model_names))
ax4.errorbar(x_pos, coef_means, 
             yerr=[np.array(coef_means) - np.array(coef_lower), 
                   np.array(coef_upper) - np.array(coef_means)],
             fmt='o', capsize=5, capthick=2, markersize=8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_names, rotation=45, ha='right')
ax4.set_ylabel('回归系数', fontsize=12)
ax4.set_title('回归系数置信区间比较', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('final_visualizations/模型稳定性分析.png', dpi=300, bbox_inches='tight')
plt.close()

print("回归分析图表生成完成！")
