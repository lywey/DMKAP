from __future__ import print_function
import pickle
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from graph import GraphConvolution
from utils import *


# Define parameters
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 30  # early stopping patience


# Get data
# X:features  A:graph  y:labels
X = np.eye(5608, dtype=np.int32)

adjFile = '../data/mimic_gcn.adj'
A = pickle.load(open(adjFile, 'rb'))

y = np.load('../data/mimic_gcnlabel.npy')
y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(y)


# Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
print('Using local pooling filters...')
A_ = preprocess_adj(A, SYM_NORM)
support = 1
graph = [X, A_]
G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]


X_in = Input(shape=(X.shape[1],))
E = Dense(128, name='gcn_emb')(X_in)

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H = Dropout(rate=0.5)(E)
H = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(rate=0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), weighted_metrics=['acc'])
model.summary()

# Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

# Train
validation_data = (graph, y_val, val_mask)
model.fit(graph, y_train, sample_weight=train_mask,
          batch_size=A.shape[0],
          epochs=NB_EPOCH,
          verbose=1,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[es_callback])

# Evaluate model on the test data
eval_results = model.evaluate(graph,y_test,
                              sample_weight=test_mask,
                              batch_size=A.shape[0])
print('Test Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

w_dense = model.get_layer('gcn_emb').get_weights()
pickle.dump(w_dense, open('../data/gcn_emb' + '.emb', 'wb'), -1)
