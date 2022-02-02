from hashlib import new
import tensorflow as tf
import resource

class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print("[MemoryCallback]: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class LearningRateReducer(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    if (epoch % 100):
        new_lr = old_lr * tf.math.exp(-0.1)
    else:
        new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)