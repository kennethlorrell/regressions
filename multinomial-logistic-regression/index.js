require('@tensorflow/tfjs-node');
const loadCSV = require('../utils/load-csv');
const MultinomialLogisticRegression = require('./MultinomialLogisticRegression');
const plot = require('node-remote-plot');
const _ = require('lodash');

const { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'horsepower',
    'displacement',
    'weight'
  ],
  labelColumns: [
    'mpg'
  ],
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);

      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      }

      return [0, 1, 0];
    }
  }
});

const regression = new MultinomialLogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5
});

regression.train();

regression.predict([[215, 440, 2.16]]).print();
// console.log(regression.test(testFeatures, testLabels));
