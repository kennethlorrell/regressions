require('@tensorflow/tfjs-node');
const MultinomialLogisticRegression = require('./MultinomialLogisticRegression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const loadData = () => {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map((feature) => _.flatMap(feature));
  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = Array.from({ length: 10 }).fill(0)
    row[label] = 1;

    return row;
  });

  return {
    features,
    labels: encodedLabels
  }
}

const { features, labels } = loadData();

const regression = new MultinomialLogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 50,
  batchSize: 500
});

regression.train();

const testMnistData = mnist.testing(0, 10000);

const testFeatures = testMnistData.images.values.map((feature) => _.flatMap(feature));
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = Array.from({ length: 10 }).fill(0)
  row[label] = 1;

  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log(`Accuracy is ${accuracy}`);

plot({
  x: regression.costHistory.reverse()
});
