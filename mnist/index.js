require('@tensorflow/tfjs-node');
const MultinomialLogisticRegression = require('./MultinomialLogisticRegression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const TRAINING_SIZE = 1000;
const TEST_SIZE = TRAINING_SIZE / 10;

const mnistData = mnist.training(0, TRAINING_SIZE);

const features = mnistData.images.values.map((feature) => _.flatMap(feature));
const encodedLabels = mnistData.labels.values.map((label) => {
  const row = Array.from({ length: 10 }).fill(0)
  row[label] = 1;

  return row;
});

const regression = new MultinomialLogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 5,
  batchSize: 100
});

regression.train();

const testMnistData = mnist.testing(0, TEST_SIZE);

const testFeatures = testMnistData.images.values.map((feature) => _.flatMap(feature));
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = Array.from({ length: 10 }).fill(0)
  row[label] = 1;

  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log(`Accuracy is ${accuracy}`);
