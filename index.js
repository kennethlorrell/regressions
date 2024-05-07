require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./LinearRegression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'displacement',
    'weight',
    'horsepower'
  ],
  labelColumns: [
    'mpg'
  ]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.01,
  iterations: 100
});

regression.train();
console.log(regression.test(testFeatures, testLabels));

// console.log(`Updated M is: ${regression.weights.get(1, 0)}, updated B is: ${regression.weights.get(0, 0)}`);
