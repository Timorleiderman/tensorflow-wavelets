import OpenDVC
import numpy as np
import load
import tensorflow as tf
import matplotlib.pyplot as plt

tf.executing_eagerly()

batch_size = 1
EPOCHS = 1
Height = 240
Width = 240
Channel = 3
lr_init = 1e-4
samples = 10
I_QP=27

model = OpenDVC.OpenDVC(width=Width, height=Height, batch_size=batch_size, num_filters=128)
# model.summary()
model.compile()
print("* [Model compiled]...")
args = OpenDVC.Arguments()

# folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
# data = load.load_data_vimeo90k(samples, batch_size, Height, Width, Channel, folder, I_QP)
print("* [Loading dataset]...")
data = load.load_data_vimeo90k("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy",
                                samples, Height, Width, Channel, I_QP)

dataset = tf.data.Dataset.from_tensor_slices(data)
# data_out = tf.data.Dataset.from_tensor_slices(data_out)

# for perm in dataset:
#     print(perm.shape)

print("* [Loading model/weights]...")
# model.load_weights(args.model_checkpoints)
model.load_weights(args.model_checkpoints_me)
# model.load_weights(args.model_checkpoints_mv)
# model.load_weights(args.model_checkpoints_mc)
# model = tf.keras.models.load_model(args.model_save)

print("Going to fit")
hist = model.fit(
        dataset,
        epochs=EPOCHS, 
        verbose=1, 
        callbacks=
            [
            OpenDVC.MemoryCallback(),
            # tf.keras.callbacks.ModelCheckpoint(filepath=args.model_checkpoints, save_weights_only=True, save_freq='epoch',monitor="train_loss_MV",mode='min',  save_best_only=True, verbose=2), 
            # tf.keras.callbacks.ModelCheckpoint(filepath=args.model_checkpoints_me, save_weights_only=True, save_freq='epoch', monitor="train_loss_ME", mode='min',  save_best_only=True, verbose=2), 
            tf.keras.callbacks.ModelCheckpoint(filepath=args.model_checkpoints_mv, save_weights_only=True, save_freq='epoch',monitor="train_loss_MV",mode='min',  save_best_only=True, verbose=2), 
            # tf.keras.callbacks.ModelCheckpoint(filepath=args.model_checkpoints_mc, save_weights_only=True, save_freq='epoch',monitor="train_loss_MC",mode='min',  save_best_only=True, verbose=2), 

            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(log_dir=args.backup_restore, histogram_freq=0, update_freq="epoch"),
            tf.keras.callbacks.experimental.BackupAndRestore(args.backup_restore),
            ],

        )  

print("Print metrics  ...")
for key in hist.history:
    print(key)

print("Done Training ...")

# model.save("test_save", save_format="tf")
# model.save_weights("model.h5")

# tf.keras.models.save_model(model, args.model_save)
# tf.saved_model.save(model, args.model_save)
print("saved", args.model_save)