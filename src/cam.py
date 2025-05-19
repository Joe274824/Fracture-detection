import numpy as np
import nibabel as nib
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.ndimage import zoom
from mayavi import mlab 
from keras import backend as K

tf.keras.mixed_precision.set_global_policy('float32')
K.set_floatx('float32')

# 加载模型
model = tf.keras.models.load_model('3d_densenet_model.h5')
for layer in model.layers:
    if layer.dtype == 'float16':
        layer._dtype_policy = tf.keras.mixed_precision.Policy('float32')
# 目标尺寸
TARGET_SHAPE = (128, 128, 128)

def resize_image(image, target_shape):
    # 计算缩放因子
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(3)]
    # 使用zoom函数进行缩放
    resized_image = zoom(image, zoom_factors, order=1)  # order=1为线性插值
    return resized_image

def get_label_from_filename(filename):
    if 'fracture' in filename:
        return 1  # 骨折
    elif 'healthy' in filename:
        return 0  # 健康
    else:
        raise ValueError(f"文件名未包含有效标签: {filename}")
    

# 定义从文件夹加载NIfTI图像的函数
def load_nifti_images_from_folder(folder, batch_size):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith('.nii'):
            img_path = os.path.join(folder, filename)
            img = nib.load(img_path).get_fdata()
            img = resize_image(img, TARGET_SHAPE)  # 使用resize_image函数进行缩放
            images.append(img)

            label = get_label_from_filename(filename)
            labels.append(label)
            filenames.append(filename)
            if len(images) == batch_size:
                yield np.array(images), np.array(labels), filenames
                images, labels, filenames = [], [], [] 

    if len(images) > 0:
        yield np.array(images), np.array(labels), filenames

# 评估模型并返回真实标签和预测标签
def evaluate_model(folder, batch_size):
    y_true = []
    y_pred = []
    filenames = []
    for images_batch, labels_batch, batch_filenames in load_nifti_images_from_folder(folder, batch_size):
        images_batch = images_batch.astype(np.float32)
        predictions = model.predict(images_batch)
        predicted_classes = np.argmax(predictions, axis=1)

        y_true.extend(labels_batch)
        y_pred.extend(predicted_classes)
        filenames.extend(batch_filenames)

    return np.array(y_true), np.array(y_pred), filenames

def grad_cam(model, image, class_index):
    # 获取模型的最后一个卷积层
    last_conv_layer = model.get_layer('dense4_block16_conv2')  # 请替换为你的模型最后一个卷积层的名称
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(np.expand_dims(image, axis=0))
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0] #check this line
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 权重乘以特征图
    conv_outputs = conv_outputs[0]

    print("Predictions:", preds.numpy())
    print("Loss:", loss.numpy())
    print("Gradients:", grads.numpy())  # 打印梯度
    print("Conv outputs shape:", conv_outputs.shape)

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # 应用ReLU以去除负值
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap

def overlay_grad_cam(image, heatmap, alpha=0.4):
    image = image.astype(np.float32)
    heatmap = heatmap.astype(np.float32)
    # 将热图调整到图像大小
    heatmap_resized = zoom(heatmap, (TARGET_SHAPE[0] / heatmap.shape[0], 
                                       TARGET_SHAPE[1] / heatmap.shape[1], 
                                       TARGET_SHAPE[2] / heatmap.shape[2]), order=1)

    # 将热图应用到原图上
    overlay = image + alpha * heatmap_resized
    overlay = np.clip(overlay, 0, 1)  # 确保值在0到1之间
    return overlay

# 进行评估
folder_path = 'E://UOA/2024-s1/Research project/repo/shoulder_ct/visualisation/fracture'  # NIfTI文件的路径
y_true, y_pred, filenames = evaluate_model(folder_path, batch_size=32)  # 选择适合的batch_size

# calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Fracture'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
plt.title('Confusion Matrix')
plt.show()

def visualize_3d_cam(image, heatmap, alpha=0.4):
    if np.any(heatmap):  # 检查是否有非零值
        print("Heatmap contains valid data.")
    else:
        raise ValueError("Heatmap is empty or contains only zeros.")
    
    # 使用Mayavi显示3D结构
    mlab.figure(size=(400, 400), bgcolor=(0, 0, 0))

    # 原始CT图像显示（灰度）
    src = mlab.pipeline.scalar_field(image)
    mlab.pipeline.volume(src, vmin=image.min(), vmax=image.max())  # 使用体绘制显示CT图像

    # CAM热力图显示（使用红色）
    overlay = overlay_grad_cam(image, heatmap, alpha)  # 调用叠加函数
    cam = mlab.pipeline.scalar_field(overlay)
    mlab.pipeline.iso_surface(cam, contours=[overlay.max() * 0.5], opacity=0.5, color=(1, 0, 0))  # 半透明显示

    mlab.show()  # 显示窗口

# 计算并可视化 Grad-CAM
for i in range(len(filenames)):
    if y_pred[i] != y_true[i]:  
        image_path = os.path.join(folder_path, filenames[i])
        image = nib.load(image_path).get_fdata()
        image = resize_image(image, TARGET_SHAPE)
        print("Image shape:", image.shape)
        print("Image values:", image)
        class_index = y_pred[i]
        heatmap = grad_cam(model, image, class_index)
        print("Heatmap shape:", heatmap.shape)
        print("Heatmap values:", heatmap)
        visualize_3d_cam(image, heatmap)