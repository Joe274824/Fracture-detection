import os
import numpy as np
import tensorflow as tf
import random
import nibabel as nib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Input, Conv3D, BatchNormalization, Activation, Add, GlobalAveragePooling3D, Dense, Concatenate, AveragePooling3D
from keras.models import Model
from keras.layers import Dropout
from keras.regularizers import l2
from scipy.ndimage import zoom, rotate, shift
from tqdm.keras import TqdmCallback
from keras import mixed_precision 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import load_model

# random seed
seed_value = 21
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# GPU Randomness
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tensorboard_callback = TensorBoard(log_dir='./logs')

def load_3d_image(filepath, image_shape):
    try:
        img = nib.load(filepath).get_fdata()
        # Check if the image dimension is zero
        if img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] == 0:
            print(f"Warning: Image {filepath} has zero dimensions: {img.shape}. Skipping this image.")
            return None
        img = np.expand_dims(img, axis=-1)
        img = resize_3d_image(img, image_shape)
        return img
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None

def resize_3d_image(img, target_shape):
    current_shape = img.shape
    if 0 in current_shape:
        raise ValueError(f"Image has zero dimension: {current_shape}")
    zoom_factors = (target_shape[0] / current_shape[0], 
                    target_shape[1] / current_shape[1], 
                    target_shape[2] / current_shape[2])
    img = np.squeeze(img)
    resized_img = zoom(img, zoom_factors, order=1)
    resized_img = np.expand_dims(resized_img, axis=-1)  # Adding channel dimension
    return resized_img

def augment_image(img):

    # make sure that img is a 4D img (128, 128, 128, 1)
    assert img.shape[-1] == 1, f"Unexpected image shape: {img.shape}"

    # Only the first 3 dimensions (x, y, z) are augmented with data without changing the channel dimensions
    shift_x = np.random.uniform(-2, 2)
    shift_y = np.random.uniform(-2, 2)
    shift_z = np.random.uniform(-2, 2)
    img = img[:, :, :, 0] 
    img = shift(img, shift=[shift_x, shift_y, shift_z], mode='nearest')

    # rotate image
    rotate_angle = np.random.uniform(-5, 5)
    img = rotate(img, angle=rotate_angle, axes=(0, 1), reshape=False, mode='nearest')

    # make sure that the image is still 4D
    img = np.expand_dims(img, axis=-1) 

    # flip image
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=0)  
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=1) 

    # add noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.01, img.shape)
        img += noise

    return img

def get_CAM(model, img):
    # 获取最后一个卷积层的名称
    last_conv_layer = model.get_layer('initial_conv')  # 替换为您的最后一个卷积层名称
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img, axis=0))
        loss = predictions[:, 1]  # 选择类别 1（骨折）的预测结果

    # 计算梯度
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 生成 CAM
    conv_outputs = conv_outputs[0]  # 获取特征图
    for i in range(len(pooled_grads)):
        conv_outputs[:, :, :, i] *= pooled_grads[i]

    heatmap = tf.reduce_mean(conv_outputs, axis=-1)  # 获取通道的平均值
    heatmap = tf.maximum(heatmap, 0)  # 只保留正值
    heatmap /= tf.reduce_max(heatmap)  # 归一化到 0-1
    return heatmap.numpy()

def data_generator(image_paths, labels, batch_size, image_shape, augment=False):
    while True:
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths)) 
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = [] 
            for i in batch_indices:
                img = load_3d_image(image_paths[i], image_shape)
                if img is None:
                    continue 
                if augment:
                    img = augment_image(img)

                if img.shape != image_shape:  # Check if the image shape matches the expected shape
                    print(f"Warning: Image shape {img.shape} does not match expected shape {image_shape}. Skipping image.")
                    continue
                batch_images.append(img)
                batch_labels.append(to_categorical(labels[i], num_classes=2))
            if len(batch_images) == 0:
                continue 
            yield np.array(batch_images), np.array(batch_labels)

# data path
fracture_folder = 'E://Research Project/Hip preprocessed Data/fracture'
healthy_folder = 'E://Research Project/Hip preprocessed Data/healthy'

# image shape
image_shape = (128, 128, 128, 1) # 3D图像的目标大小

# load all files
fracture_files = [os.path.join(fracture_folder, f) for f in os.listdir(fracture_folder) if f.endswith('.nii')]
healthy_files = [os.path.join(healthy_folder, f) for f in os.listdir(healthy_folder) if f.endswith('.nii')]
all_files = fracture_files + healthy_files
all_labels = [1] * len(fracture_files) + [0] * len(healthy_files)

# shuffle
combined = list(zip(all_files, all_labels))
random.shuffle(combined)
all_files, all_labels = zip(*combined)

print("=====================")

# batch size
batch_size = 16

# train and validation split
train_split = 0.8
split_index = int(len(all_files) * train_split)
# create data generator
train_gen = data_generator(all_files[:split_index], all_labels[:split_index], batch_size, image_shape, augment=True)
val_gen = data_generator(all_files[split_index:], all_labels[split_index:], batch_size, image_shape, augment=False)

# calculate steps
train_steps = len(all_files[:split_index]) // batch_size
val_steps = len(all_files[split_index:]) // batch_size

def conv_block(x, growth_rate, name):
    inter_channel = 4 * growth_rate
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv3D(inter_channel, (1, 1, 1), padding='same', use_bias=False, name=name+'_conv1')(x1)
    
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv3D(growth_rate, (3, 3, 3), padding='same', use_bias=False, name=name+'_conv2')(x1)
    
    x = Concatenate(axis=-1)([x, x1])
    return x

def dense_block(x, blocks, growth_rate, name):
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name+'_block'+str(i+1))
    return x

def transition_block(x, reduction, name):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(int(tf.keras.backend.int_shape(x)[-1] * reduction), (1, 1, 1), padding='same', use_bias=False, name=name+'_conv')(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name=name+'_pool')(x)
    return x

def densenet3d(input_shape, growth_rate=32, reduction=0.5, block_layers=[6, 12, 24, 16]):
    inputs = Input(shape=input_shape)
    
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same', use_bias=False,  name='initial_conv')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((2, 2, 2))(x)
    
    for i, num_blocks in enumerate(block_layers):
        x = dense_block(x, num_blocks, growth_rate, name='dense'+str(i+1))
        if i != len(block_layers) - 1:
            x = transition_block(x, reduction, name='transition'+str(i+1))
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = Model(inputs, x)
    return model

input_shape = (128, 128, 128, 1) 
model = densenet3d(input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# train model
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=30,
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=1,
    callbacks=[TqdmCallback(verbose=1), lr_reducer, early_stopping, tensorboard_callback]
)

# save model
model.save('3d_densenet_model.h5')
# model = load_model('3d_densenet_model.h5', compile=False)
# print model summary
print("Training Loss and Accuracy:")
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
print(train_loss)
print(train_accuracy)

print("Validation Loss and Accuracy:")
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
print(val_loss)
print(val_accuracy)

epochs = range(1, len(train_loss) + 1)

# Plotting training and validation loss
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

def visualize_CAM(image, heatmap):
    # 使用 matplotlib 绘制热图
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, :, 0], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # 叠加热图
    plt.title('Class Activation Map')
    plt.axis('off')
    plt.show()


def load_validation_data(file_paths, labels, image_shape):
    images = []
    actual_labels = []
    for i, filepath in enumerate(file_paths):
        img = load_3d_image(filepath, image_shape)
        if img is not None:
            images.append(img)
            actual_labels.append(labels[i])
        else:
            print(f"Skipping invalid image at {filepath}")
    images = np.array(images)
    actual_labels = np.array(actual_labels)
    return images, to_categorical(actual_labels, num_classes=2)

# load validation data
val_images, val_labels = load_validation_data(all_files[split_index:], all_labels[split_index:], image_shape)

# 获取某个验证样本的 CAM
sample_index = 0  # 选择一个样本
heatmap = get_CAM(model, val_images[sample_index])
visualize_CAM(val_images[sample_index], heatmap)

# predict on validation data
val_predictions = model.predict(val_images, batch_size=8)
val_predictions = np.argmax(val_predictions, axis=-1)

# actual labels
val_actual_labels = np.argmax(val_labels, axis=-1)

# calculate confusion matrix
cm = confusion_matrix(val_actual_labels, val_predictions)

print("Confusion Matrix:")
print(cm)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Fracture'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
plt.title('Confusion Matrix')
plt.show()