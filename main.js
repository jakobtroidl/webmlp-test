import { createMLP, from_tfjs } from "web-mlp";

async function testMLP() {
  let batch_size = 70000;
  let tile_size = 8; // must not be bigger than 16

  const path =
    "https://jakobtroidl.github.io/data/trainedModelOriginal/model.json"; // path to tensorflow.js model

  let tfjs_model = await from_tfjs(path); // load tfjs model
  let model = await createMLP(tfjs_model, batch_size, tile_size); // convert model for fast inference
  let X = Float32Array.from(Array(batch_size * model.inputSize).fill(0), () =>
    Math.random()
  ); // generate random a input

  let result = await model.inference(X); // inference the model
  console.log("result", result); // plot predictions
}

testMLP();
