import matplotlib.pyplot as plt
 
# 创建一个图像，并添加两个子图
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)  # 表示2行1列的子图布局，这是第1个子图
ax2 = fig.add_subplot(2, 1, 2)  # 这是第2个子图
 
# 在子图上绘制数据
ax1.plot([1, 2, 3], [4, 5, 6])
ax2.plot([1, 2, 3], [6, 5, 4])
 
# 显示图像
plt.show()