require('@tensorflow/tfjs-node');
const loadCSV = require('../utils/load-csv');
const LinearRegression = require('./LinearRegression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'horsepower',
    'displacement',
    'weight'
  ],
  labelColumns: [
    'mpg'
  ]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();

// plot({
//   x: regression.mseHistory.reverse(),
//   xLabel: 'Iteration #',
//   yLabel: 'Mean Squared Error'
// });

console.log(regression.test(testFeatures, testLabels));

regression.predict([
  [120, 2, 380]
]).print();
