import tensorflow as tf
from u_net import U_Net
from preprocessing import process_data
import numpy as np
import random
import time
import os
from matplotlib import pyplot as plt


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        # low learning rate to compensate for low batch size (normally 1e-4)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.batch_size = 6

        self.u_net = U_Net(12)

    def call(self, image_data):

        image_data = self.u_net(tf.math.minimum(image_data, 1.))

        return tf.math.minimum(tf.nn.depth_to_space(image_data, 2), 1.)

    def loss(self, pred_image, ground_truth):

        return tf.reduce_sum(tf.math.abs(pred_image-ground_truth))

    def accuracy(self, pred_image, ground_truth):

        ssim = tf.image.ssim(pred_image, ground_truth, 1)
        psnr = tf.image.psnr(pred_image, ground_truth, 1)
        return psnr, ssim


def train(model, in_dataset, gt_dataset):
    """
    This is the function that trains the data. We iterate over ever input image batch and reference image batch
    :param model: The model to train
    :param in_dataset: iterator over the input data
    :param gt_dataset: iterator over the reference data
    :return: list of losses for each batch
    """
    losses = []
    i = 0
    for in_images in in_dataset:
        i += 1
        gt_images = gt_dataset.next()

        # crop out random 512 x 512 patch
        patch_dim = 512

        max_row = tf.shape(in_images)[1].numpy() - patch_dim
        max_col = tf.shape(in_images)[2].numpy() - patch_dim

        row_num = random.randint(0, max_row)
        col_num = random.randint(0, max_col)

        in_images = in_images[:, row_num:row_num+patch_dim, col_num:col_num+patch_dim, :]
        gt_images = gt_images[:, row_num*2:(row_num+patch_dim)*2, col_num*2:(col_num+patch_dim)*2, :]

        # rotate a random number of times
        rot_num = int(random.random()*4)
        in_images = tf.image.rot90(in_images, rot_num)
        gt_images = tf.image.rot90(gt_images, rot_num)

        with tf.GradientTape() as tape:
            pred_images = model.call(in_images)
            loss = model.loss(pred_images, gt_images)
            losses.append(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"completed batch #{i}")
        if i % 20 == 0:
            pred_images = model.call(in_images)
            val_psnr, val_ssim = model.accuracy(pred_images, gt_images)
            print(f"After batch {i}, PSNR: {val_psnr}, SSIM: {val_ssim}")

    return losses


def test(model, in_dataset, gt_dataset):
    """
    This tests the model, produces image results and returns PSNR and SSIM
    :param model: model to test
    :param in_dataset: input dataset as an iterator
    :param gt_dataset: ground-truth dataset as an iterator
    :return: PSNR, SSIM
    """
    ssim_vals = []
    psnr_vals = []
    i = 0
    for in_images in in_dataset:
        i += 1
        gt_images = gt_dataset.next()
        pred_images = model.call(in_images)

        psnr, ssim = model.accuracy(pred_images, gt_images)
        ssim_vals.append(ssim)
        psnr_vals.append(psnr)

        save_dir = "tmp\\testing_results"
        img_name = f"testing_img_num_{i}.jpg"
        tf.keras.preprocessing.image.save_img(os.path.join(save_dir,img_name), pred_images[0])
    return tf.reduce_mean(psnr_vals), tf.reduce_mean(ssim_vals)


def main():
    """
    This fetches the data, runs the model and tests the model
    :return:
    """

    model = Model()

    data_path = "F:\\Final Project Data\\Sony"
    train_file = "Sony_train_list.txt"
    test_file = "Sony_test_list.txt"
    val_file = "Sony_val_list.txt"  # filename for validation dataset

    train_in_images, train_gt_images = process_data(data_path, train_file, model.batch_size)
    test_in_images, test_gt_images = process_data(data_path, val_file, 1)  # using the validation data for brevity
    val_in_images, val_gt_images = process_data(data_path, val_file, 1)

    epochs = 4
    losses = []
    for i in range(epochs):
        start_time = time.time()
        losses = losses + train(model, train_in_images.as_numpy_iterator(), train_gt_images.as_numpy_iterator())
        end_time = time.time()
        print(f"Completed epoch #{i+1} after {end_time-start_time} seconds")
        if i+1 % 100 == 0:
            val_psnr, val_ssim = test(model, val_in_images.as_numpy_iterator(), val_gt_images.as_numpy_iterator())
            print(f"After epoch {i}, PSNR: {val_psnr}, SSIM: {val_ssim}")
    graph_losses(losses)
    psnr, ssim = test(model, test_in_images.as_numpy_iterator(), test_gt_images.as_numpy_iterator())
    tf.keras.models.save_model(model, "tmp\\training_checkpoints")
    print(f"Mean PSNR: {psnr.numpy()}")
    print(f"Mean SSIM: {ssim.numpy()}")


def graph_losses(losses):
    plt.plot(losses, '-o')
    plt.ylabel('L1 loss')
    plt.xlabel('training batch')
    plt.show()


if __name__ == '__main__':
    main()

