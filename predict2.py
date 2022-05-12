model = tf.keras.models.load_model(
    "best-model-370b-siamese-0", 
    custom_objects={"loss1_rank_triplet": loss1_rank_triplet, "loss2_mse_target": loss2_mae_target})

model_scoring = model.layers[2]
model_scoring.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4,  # Karpathy, 2019
        beta_1=.9, beta_2=.999, epsilon=1e-7,  # Kingma and Ba, 2014, p.2
        amsgrad=True  # Reddi et al, 2018, p.5-6
    ),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error'])

xinputs = np.hstack([feats1, feats2, feats3, feats4, feats5, feats6])
repr, y1hat, y2hat, y3hat = model_scoring.predict(xinputs)

loss = tf.math.reduce_mean(tf.math.pow(y1hat - y1mos, 2)) \
    + tf.math.reduce_mean(tf.math.pow(y2hat - y2mos, 2)) \
    + tf.math.reduce_mean(tf.math.pow(y3hat - y3mos, 2))
loss
