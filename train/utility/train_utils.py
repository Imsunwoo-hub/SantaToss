import tensorflow as tf


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, loss_object, attention):
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            context_vector, _ = attention(features, hidden)

            predictions, hidden = decoder(dec_input, context_vector)

            loss += loss_function(target[:, i], predictions, loss_object)

            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def validation_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, loss_object, attention):
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
        context_vector, _ = attention(features, hidden)

        predictions, hidden = decoder(dec_input, context_vector)

        loss += loss_function(target[:, i], predictions, loss_object)

        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss


def evaluate(image, image_features_extract_model, encoder, bahdanau_attention, decoder, max_length, tokenizer):
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(preprocessing.load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([2], 0)
    result = []

    for i in range(max_length):
        context_vector, attention_weights = bahdanau_attention(features, hidden)

        predictions, hidden = decoder(dec_input, context_vector)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        if tokenizer[predicted_id] == '<end>':
            return result
        if not tokenizer[predicted_id] == '<unk>':
            result.append(tokenizer[predicted_id])

        dec_input = tf.expand_dims([predicted_id], 0)

    return result