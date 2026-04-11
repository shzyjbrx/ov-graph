import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. 指定本地字体绝对路径
font_path = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/RGIPCOL2/SIMSUN.TTC"

# 2. 注册字体并设置为全局默认字体
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =========================
# 实验数据 1：RGCN传播层数 L
# =========================
L_values = [1, 2, 3, 4, 5]
hm_L  = [32.85, 33.62, 34.76, 33.58, 32.42]
auc_L = [16.12, 16.65, 17.23, 16.27, 16.15]

# =========================
# 打印当前实验结果
# =========================
print("RGCN传播层数敏感性实验结果：")
for l, hm, auc in zip(L_values, hm_L, auc_L):
    print(f"L={l}: HM={hm:.2f}, AUC={auc:.2f}")

# =========================
# 图1：RGCN传播层数 L 的影响
# =========================
fig1, ax1 = plt.subplots(figsize=(8, 5))

color1 = '#1f77b4'  # 学术蓝
ax1.set_xlabel('RGCN 传播层数 (L)', fontsize=12)
ax1.set_ylabel('调和平均值 (HM) %', color=color1, fontsize=12)
ax1.plot(L_values, hm_L, marker='o', color=color1, label='HM',
         linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(L_values)
ax1.set_ylim(31.0, 35.5)  # 修正：根据新数据放宽了HM轴的显示范围

ax1_twin = ax1.twinx()
color2 = '#ff7f0e'  # 学术橙
ax1_twin.set_ylabel('曲线下面积 (AUC) %', color=color2, fontsize=12)
ax1_twin.plot(L_values, auc_L, marker='s', color=color2, label='AUC',
              linewidth=2, linestyle='--', markersize=8)
ax1_twin.tick_params(axis='y', labelcolor=color2)
ax1_twin.set_ylim(14.0, 18.0)  # 修正：根据新数据放宽了AUC轴的显示范围

# plt.title('RGCN传播层数对模型性能的影响 (MIT-States)', fontsize=14)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center')

ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
fig1.savefig('rgcn_layers_analysis.png', dpi=600, bbox_inches='tight')
plt.close(fig1)

print("\n图表已生成：")
print("1. rgcn_layers_analysis.png")
print("并使用了指定的 SIMSUN.TTC 字体。")