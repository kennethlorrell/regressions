require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./LinearRegression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'horsepower'
  ],
  labelColumns: [
    'mpg'
  ]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100
});

regression.train();

console.log(`Updated m is: ${regression.m}, updated b is: ${regression.b}`);
