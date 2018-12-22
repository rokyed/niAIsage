const tf = require('@tensorflow/tfjs');

// Load the binding:
require('@tensorflow/tfjs-node-gpu');  // Use '@tensorflow/tfjs-node-gpu' if running with GPU.


// Train a simple model:
const model = tf.sequential();
model.add(tf.layers.dense({units: 1000, activation: 'relu', inputShape: [100]}));
model.add(tf.layers.dense({units: 10, activation: 'relu'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([1000, 100]);
const ys = tf.randomNormal([1000, 10]);

model.fit(xs, ys, {
  epochs: 100000,
  callbacks: {
    onEpochEnd: async (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`);
    }
  }
});

