import * as tf from '@tensorflow/tfjs';

// https://js.tensorflow.org/tutorials/core-concepts.html
async function tutorial1() {
  const shape = [2,3];
  const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
  a.print();

  const b = tf.tensor([[1.1, 2.0, 3.0], [10.0, 20.0, 30.0]]);
  b.print();

  const c = tf.tensor1d([1.2, 2.0, 3.0, 10.0, 20.0, 30.0]);
  c.print();


  const initialValues = tf.zeros([5]);
  const biases = tf.variable(initialValues);
  biases.print();

  const updatedValues = tf.tensor1d([1,2,3,4,5]);
  biases.assign(updatedValues);
  biases.print();

  a.square().print();
  a.add(b).print();

  function predict(input: number) {
    return tf.tidy(() => {
      const x = tf.scalar(input);
      const ax2 = aa.mul(x.square());
      const bx = bb.mul(x);
      const y = ax2.add(bx).add(cc);
      return y;
    });
  }

  const aa = tf.scalar(2);
  const bb = tf.scalar(4);
  const cc = tf.scalar(8);
  const result = predict(2);
  result.print();


  const data = tf.tensor1d([-1,-2,-3,-4,-5,1,23,2,3,4,5]);
  const labels = tf.tensor1d([-1,-1,-1,-1,-1,1,1,1,1,1,1]);
  const LEARNING_RATE = 0.05;

  const model = tf.sequential();
  model.add(
    tf.layers.simpleRNN({
      units: 20,
      recurrentInitializer: 'GlorotNormal',
      inputShape: [11, 3]
    })
  );

  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({optimizer, loss: 'categoricalCrossentropy'});
  await model.fit(data, labels);
  const prediction = <tf.Tensor<tf.Rank>>model.predict(tf.scalar(1));
  prediction.print();
}
tutorial1();