import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

class HessianDropout(layers.Layer):


    def __init__(self, base_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_rate = base_rate
        self.drop_probs = None      
        self.last_mask = None      

    def set_drop_probs(self, probs):        
        self.drop_probs = tf.convert_to_tensor(probs, dtype=tf.float32)

    def clear_drop_probs(self):
        self.drop_probs = None

    def call(self, inputs, training=None):
        if not training:
            return inputs

        batch = tf.shape(inputs)[0]
        units = tf.shape(inputs)[1]

        if self.drop_probs is not None:            
            probs = tf.broadcast_to(self.drop_probs[None, :], (batch, units))
        else:
            probs = tf.fill((batch, units), tf.cast(self.base_rate, tf.float32))
        
        keep_prob = 1.0 - probs
        mask = tf.cast(tf.random.uniform((batch, units)) < keep_prob, tf.float32)
        self.last_mask = mask

        
        scale = 1.0 / tf.maximum(tf.reduce_mean(keep_prob), 1e-8)
        return inputs * mask * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"base_rate": self.base_rate})
        return cfg
        
        



class HessianDropoutTrainer:
    
    def __init__(self, model, optimizer, loss_fn,
                 alpha=1.0, beta=-1.0, eps=1e-8, m_probes=1,
                 warmup_epochs=10, base_rate=0.5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.m_probes = m_probes
        self.warmup_epochs = warmup_epochs
        self.base_rate = base_rate
        
        self.pairs = self._find_pairs(model)
        print(f"[HessianDropoutTrainer] pairs found: "
              f"{[(d, h) for d, h in self.pairs]}")

    @staticmethod
    def _find_pairs(model):        
        pairs = []
        for layer in model.layers:
            if isinstance(layer, HessianDropout):                
                try:
                    inp = layer.input

                    producer = inp._keras_history[0]
                    if isinstance(producer, layers.Dense):
                        pairs.append((producer.name, layer.name))
                except Exception as e:
                    print("skip:", layer.name, e)
        return pairs

    def _hvp_trace(self, dense_layer, x_batch, y_batch):
        W = dense_layer.kernel
        estimates = []
        for _ in range(self.m_probes):
            nu = tf.cast(tf.random.uniform(tf.shape(W)) > 0.5, tf.float32) * 2.0 - 1.0
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:
                    tape1.watch(W)
                    preds = self.model(x_batch, training=False)
                    loss = self.loss_fn(y_batch, preds)
                g = tape1.gradient(loss, W)
                if g is None:
                    continue
                gv = tf.reduce_sum(g * nu)
            Hv = tape2.gradient(gv, W)
            if Hv is None:
                continue
            est = tf.reduce_sum(nu * Hv)
            estimates.append(tf.maximum(est, 0.0).numpy())
        return float(np.mean(estimates)) if estimates else 0.0

    def _wsr(self, dense_layer, x_batch, y_batch, c_hat):
        W = dense_layer.kernel
        with tf.GradientTape() as tape:
            preds = self.model(x_batch, training=False)
            loss = self.loss_fn(y_batch, preds)
        g = tape.gradient(loss, W)
        if g is None:
            return np.zeros(W.shape[1], dtype=np.float32)
        g_norm = tf.norm(g, axis=0) + self.eps
        w_norm = tf.norm(W, axis=0) + self.eps
        return ((g_norm / w_norm) * c_hat).numpy()

    def _wsr_to_prob(self, wsr):
        return 1.0 / (1.0 + np.exp(-(self.alpha * wsr + self.beta)))

    # ---------- fit ----------
    def fit(self, x_train, y_train, epochs, batch_size=64,
            validation_data=None, verbose=1):
        n = len(x_train)
        n_batches = n // batch_size
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            x_shuf, y_shuf = x_train[idx], y_train[idx]
            total_loss = 0.0

            for b in range(n_batches):
                xb = tf.constant(x_shuf[b*batch_size:(b+1)*batch_size])
                yb = tf.constant(y_shuf[b*batch_size:(b+1)*batch_size])

                in_warmup = epoch < self.warmup_epochs

                if not in_warmup:
                    # مرحله WSR برای هر جفت
                    for dense_name, hdrop_name in self.pairs:
                        dense_layer = self.model.get_layer(dense_name)
                        hdrop = self.model.get_layer(hdrop_name)
                        c_hat = self._hvp_trace(dense_layer, xb, yb)
                        wsr = self._wsr(dense_layer, xb, yb, c_hat)
                        if np.var(wsr) < 1e-8:
                            probs = np.full_like(wsr, self.base_rate)
                        else:
                            probs = self._wsr_to_prob(wsr)
                        hdrop.set_drop_probs(probs)
                else:
                    
                    for _, hdrop_name in self.pairs:
                        self.model.get_layer(hdrop_name).clear_drop_probs()

                
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = self.loss_fn(yb, preds)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))
                total_loss += float(loss.numpy())

            msg = f"Epoch {epoch+1:03d} | loss={total_loss/n_batches:.4f}"
            if validation_data is not None:
                xv, yv = validation_data
                _, acc = self.model.evaluate(xv, yv, verbose=0)
                msg += f" | val_acc={acc:.4f}"
            if verbose:
                print(msg)
        return self        