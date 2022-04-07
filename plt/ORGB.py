import matplotlib.pyplot as plt

def orgb(img):

	plt.figure(num='astronaut',figsize=(8,8))  # 创建一个名为astronaut的窗口,并设置大小

	plt.subplot(2,2,1)          # 将窗口分为两行两列四个子图，则可显示四幅图片
	plt.title('origin image')   # 第一幅图片标题
	plt.imshow(img)             # 绘制第一幅图片

	plt.subplot(2,2,2)          # 第二个子图
	plt.title('R channel')      # 第二幅图片标题
	plt.imshow(img[:,:,0],plt.cm.gray)      # 绘制第二幅图片,且为灰度图
	plt.axis('off')             # 不显示坐标尺寸

	plt.subplot(2,2,3)          # 第三个子图
	plt.title('G channel')      # 第三幅图片标题
	plt.imshow(img[:,:,1],plt.cm.gray)      # 绘制第三幅图片,且为灰度图
	plt.axis('off')             # 不显示坐标尺寸

	plt.subplot(2,2,4)          # 第四个子图
	plt.title('B channel')      # 第四幅图片标题
	plt.imshow(img[:,:,2],plt.cm.gray)      # 绘制第四幅图片,且为灰度图
	plt.axis('off')             # 不显示坐标尺寸

	plt.show()   # 显示窗口