import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from utils import *


# Define parameters
GRU_DIM = 128
TRAIN_EPOCH = 100
BATCH_SIZE = 100

# The gpu is allocated according to the memory required
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ScaledDotProductAttention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(2, input_shape[0][2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1][2], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        super(ScaledDotProductAttention, self).build(input_shape)

    def call(self, inputs):
        Lt, rnn_ht = inputs
        WQ = K.dot(rnn_ht, self.W)
        WK = K.dot(Lt, self.kernel[0])
        WV = K.dot(Lt, self.kernel[1])

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (128 ** 0.5)

        weights = K.softmax(QK)
        context_vector = K.batch_dot(weights, WV)

        return context_vector, weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


if __name__ == '__main__':
    seqFile = './build_trees/mimic.seqs'
    knowledgeFile = './data/mimic_trees.seqs'
    labelFile = './data/mimic.labels'
    gcn_emb = pickle.load(open('./data/gcn_emb.emb', 'rb'))

    model_file = './models/DMKAP_default'

    # gcn Embedding
    diagcode_emb = gcn_emb[0][:ICD_NUM]
    knowledge_emb = gcn_emb[0][ICD_NUM:]

    train_set, valid_set, test_set = load_data(seqFile, labelFile, knowledgeFile)
    x, y, tree = padMatrix(train_set[0], train_set[1],train_set[2])
    x_valid, y_valid, tree_valid = padMatrix(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    model_input = keras.layers.Input((x.shape[1], x.shape[2]), name='model_input')
    mask = keras.layers.Masking(mask_value=0)(model_input)
    emb = keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')(mask)
    rnn = keras.layers.Bidirectional(keras.layers.GRU(GRU_DIM, return_sequences=True, dropout=0.5))(emb)
    head1 = keras.layers.Attention()([rnn, rnn])
    head2 = keras.layers.Attention()([rnn, rnn])


    tree_input = keras.layers.Input((tree.shape[1], tree.shape[2]), name='tree_input')
    tree_mask = keras.layers.Masking(mask_value=0)(tree_input)
    tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),
                                    trainable=True, name='tree_emb')(tree_mask)

    head1, weights1 = ScaledDotProductAttention(output_dim=128)([tree_emb, head1])
    head2, weights2 = ScaledDotProductAttention(output_dim=128)([tree_emb, head2])

    st = keras.layers.concatenate([head1, head2], axis=-1)
    model_output = keras.layers.Dense(LABEL_NUM, activation='softmax', name='main_output')(st)

    model = keras.models.Model(inputs=[model_input, tree_input], outputs=model_output)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=0.001), loss='binary_crossentropy')

    checkpoint = keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, mode='auto')

    print("Start Train")
    time1 = time.time()
    history = model.fit([x, tree], y,
                        epochs=TRAIN_EPOCH,
                        batch_size=BATCH_SIZE,
                        validation_data=([x_valid, tree_valid], y_valid),
                        callbacks=[checkpoint])
    print("The computing time cost is {:.4f}s".format(time.time()-time1))

    model = keras.models.load_model(model_file)

    # evaluation
    res = model.predict([x_test, tree_test])
    y_pred = convert2preds(res)
    y_true = process_label(test_set[1])
    codeLevelRes = code_level_accuracy(y_true, y_pred)
    visitLevelRes = visit_level_precision(y_true, y_pred)
    print("start evaluate")
    for k in [5, 10, 15, 20]:
        print("code-level accuracy@{}：{}   visit-level precision@{}：{}".format(
                    k, codeLevelRes[int((k/5)-1)], k, visitLevelRes[int((k/5)-1)]))
