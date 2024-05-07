require('@tensorflow/tfjs-node');
const loadCSV = require('../utils/load-csv');
const LogisticRegression = require('./LogisticRegression');

const { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'horsepower',
    'displacement',
    'weight'
  ],
  labelColumns: [
    'passedemissions'
  ],
  converters: {
    passedemissions: (value) => value === 'TRUE' ? 1 : 0
  }
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50
});

regression.train();
regression.predict([
  [130, 307, 1.75],
  [88, 97, 1.065]
]).print();
