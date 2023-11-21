import os
import numpy as np
import networkx as nx
import tensorflow as tf
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam,Nadam,SGD,RMSprop,Adagrad
from sklearn.metrics import precision_score, recall_score,accuracy_score

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj

import matplotlib.pyplot as plt
import pandas as pd
################################################################################
# Config
################################################################################
learning_rate = 1e-2  # Learning rate
epochs = 250  # Number of training epochs
es_patience = 300  # Patience for early stopping
batch_size = 64  # Batch size


################################################################################
# Load data
################################################################################

def read_data(folderPath):
    output = []
    classification_map = {
        "benign":0,
        "defacement":1,
        "malware":1,
        "phishing":1,
    }
    files = os.listdir(folderPath)
    for fname in files:
        try:
            file = open(os.path.join(folderPath,fname),"r")
            url = file.readline()
            utype = file.readline()
            url = url.split('\n')[0]
            utype = utype.split('\n')[0]
            n,e = map(int,file.readline().split())

            # Node feature matrix
            x = np.zeros((n,30))

            # Adjacency Matrix
            a = np.zeros((n,n))
            
            # Truth value
            y = np.zeros((2,))
            for i in range(e):
                p,q = map(int,file.readline().split())
                a[p][q] = 1
            G = nx.from_numpy_array(a)
            tr =nx.triangles(G)
            cn = nx.core_number(G)
            sc = nx.subgraph_centrality(G)
            for i in range(n):
                featurevec = list(map(float,file.readline().split()))[:27]
                
                # Number of triangles for each node
                featurevec.append(tr[i])

                # Core Number for each node
                featurevec.append(cn[i])

                # Subgraph Centrality of each node
                featurevec.append(sc[i])
                for j in range(len(featurevec)):
                    x[i][j] = featurevec[j]
            y[classification_map[utype]] = 1
            file.close()
            output.append(Graph(x=x, a=a, y=y,name=url))
        except ValueError:
            continue
    return output


class CustomData(Dataset):
    def __init__(self, trainfolder, testfolder, **kwargs):
        self.trainfolder = trainfolder
        self.testfolder = testfolder
        super().__init__(**kwargs)

    def read(self):
        op = []
        op += read_data(self.trainfolder)
        op += read_data(self.testfolder)
        return op


data = CustomData("train","test", transforms=NormalizeAdj())
inds = [i for i in range(len(os.listdir("train")))]
data2 = data[inds]
data_te = data[[i for i in range(len(os.listdir("train")),len(data))]]
# Train/valid/test split
idxs = np.random.permutation(len(data2))
split_va = int(0.9 * len(data2))
idx_tr, idx_va = np.split(idxs, [split_va])
data_tr = data2[idx_tr]
data_va = data2[idx_va]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)


################################################################################
# Build model
################################################################################
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(128, activation="relu")
        self.conv2 = GCSConv(64, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="softmax")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output


model = Net()
optimizer = Adam(learning_rate=learning_rate)
# optimizer = Nadam(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()


################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc

def getOptName(model):
    return str(model).split()[0].split('.')[-1]


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            # print("Output Prediction", output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
training_info = [[],[],[],[],[]]
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )
        data = (epoch, *np.mean(results, 0), val_loss, val_acc)
        training_info[0].append(data[0])
        training_info[1].append(data[1])
        training_info[2].append(data[2])
        training_info[3].append(data[3])
        training_info[4].append(data[4])

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

fig, ax = plt.subplots()
ax.plot(training_info[0],training_info[1],label="Loss")
ax.plot(training_info[0],training_info[2],label="Accuracy")
ax.plot(training_info[0],training_info[3],label="Validation Loss")
ax.plot(training_info[0],training_info[4],label="Validation Accuracy")
ax.legend()
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Value")
ax.set_title("Training Data")
# plt.show()
plt.savefig("TrainingData.png")
plt.close(fig)

################################################################################
# Evaluate model
################################################################################
model.set_weights(best_weights)  # Load best model
test_loss, test_acckeras = evaluate(loader_te)

print("Done. Test loss: {:.4f}. Test accuracy: {:.2f}".format(test_loss, test_acckeras))
df = pd.read_csv("paramterTuning.csv")
df.loc[len(df.index)] = [len(data_tr),len(data_te),learning_rate,epochs,es_patience,batch_size,getOptName(optimizer),training_info[1][-1],training_info[2][-1],training_info[3][-1],training_info[4][-1],test_loss,test_acckeras]
df.to_csv("paramterTuning.csv",index=False)