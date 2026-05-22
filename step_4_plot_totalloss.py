import json
import matplotlib
# 强制设置后端，解决不出图问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp

# 你的日志文件路径
# log_path = "/mnt/qh2-nas3/00-model/00-wrs/dinov3_workspace/output_eurosat_result_swin/training_metrics.json"
# log_path = "/mnt/ht2_nas2/00-model/00-wrs/dinov3_workspace/output_eurosat_result_swin/training_metrics.json"
# log_path = "/mnt/qh2-nas3/00-model/00-wrs/dinov3_workspace/code_revised/dinov3-swin/training_metrics_swin_lr2e-4_10000.json"
# log_path = "/mnt/ht2_nas2/00-model/00-wrs/dinov3_workspace/output_eurosat_result_v2/training_metrics.json"
# log_path = "/mnt/qh2-nas3/00-model/00-wrs/dinov3_workspace/output_eurosat_result_swin_ibot_maskembstd_5e-4/training_metrics.json"
log_path = "/mnt/qh2-nas3/00-model/00-wrs/dinov3_workspace/output_eurosat_result_ibot/training_metrics.json"

# 读取日志
with open(log_path, "r") as f:
    lines = f.readlines()
    data = [json.loads(line.strip()) for line in lines if line.strip()]

loss_key = "total_loss"
# 提取数据
iterations = [d["iteration"] for d in data]
target_loss = [d[loss_key] for d in data]

# 画图
plt.figure(figsize=(12, 6))
plt.plot(iterations, target_loss, linewidth=2, color='#ff5733', label='Total Loss')
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Total Loss", fontsize=12)
plt.title("DINOv3 Training Total Loss", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

save_name = f'curve_{loss_key}_' + osp.dirname(log_path).split('/')[-1] + '.png'
save_name = osp.join(osp.dirname(log_path), save_name)
# 关键：直接保存成图片，不依赖窗口
plt.savefig(save_name, dpi=300, bbox_inches='tight')
# plt.savefig("/mnt/ht2_nas2/00-model/00-wrs/dinov3_workspace/output_eurosat_result_swin/loss_curve_swin_lr2e-4_new.png", dpi=300, bbox_inches='tight')
print("✅ 图片已保存为：loss_curve.png")

# 关闭图像释放内存
plt.close()
