require('@tensorflow/tfjs-node-gpu');  // Use '@tensorflow/tfjs-node-gpu' if running with GPU.
const tf = require('@tensorflow/tfjs');


async function bla() {
const model = tf.sequential();

console.log(tf.getBackend())

model.add(tf.layers.dense({
	units: 8,
	inputShape: 2,
	activation: 'tanh'
}));
model.add(tf.layers.dense({
	units: 1,
	activation: 'sigmoid'
}));
model.compile({
	optimizer: 'sgd',
	loss: 'meanSquaredError',
	lr: 0.6
});
// Creating dataset
const xs = tf.tensor2d([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
]);
xs.print();
const ys = tf.tensor2d([
	[0],
	[1],
	[1],
	[0]
]);
ys.print();
// Train the model
await model.fit(xs, ys, {
	batchSize: 1,
	epochs: 50000
});

}

bla()
