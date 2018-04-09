<template>
    <div class="container">

        <section class="section">
            <h1 class="title">
                {{ message }}
            </h1>
            <h2 class="subtitle">
                <p class="lead">
                    {{ `Batch: ${batchNumber}`}}
                </p>
            </h2>
        </section>

        <section class="section">
            <div class="columns">
                <div class="column">
                    <div class="canvases">
                        <div class="label" id="loss-label">
                            <p class="lead">{{ `Last loss: ${lossValues[lossValues.length - 1].loss.toFixed(2)}`}}</p>
                        </div>
                        <div id="lossCanvas"></div>
                    </div>

                </div>
                <div class="column">
                    <div class="canvases">
                        <div class="label" id="accuracy-label">
                            <p class="lead">{{ `Last accuracy: ${(accuracyValues[accuracyValues.length - 1].accuracy *
                                100).toFixed(2)}`}}</p>
                        </div>
                        <div id="accuracyCanvas"></div>
                    </div>
                </div>
                <div class="column"></div>
                <div class="column"></div>
            </div>
        </section>
        <section class="section">
            <div id="images"></div>
        </section>
    </div>
</template>


<script>
    import * as tf from '@tensorflow/tfjs'
    import {MnistData} from '../data.js'
    import embed from 'vega-embed'


    const model = tf.sequential()
    let mnistData

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelIniliazer: 'varianceScaling'
    }))

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelIniliazer: 'varianceScaling'
    }))
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))
    model.add(tf.layers.flatten())
    model.add(tf.layers.dense({
        units: 10, kernelIniliazer: 'varianceScaling', activation: 'softmax'
    }))

    const LEARNING_RATE = 0.5
    const optimizer = tf.train.sgd(LEARNING_RATE)
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']

    });

    const BATCH_SIZE = 64
    const TRAIN_BATCHES = 150

    const TEST_BATCH_SIZE = 1000
    const TEST_ITERATION_FREQUENCY = 5

    let data
    export default {


        created() {
            console.log(this.valuesCreated)
            this.mnist();
        },

        mounted() {
            this.message = 'Loading data ...'
        },

        data() {
            return {
                message: 'Loading data ...',
                lossCanvas: 'lossCanvas',
                accuracyCanvas: 'accuracyCanvas',
                lossValues: [],
                accuracyValues: [],
                lossCreated: false,
                accuracyCreated: false,
                batchNumber: null,
                predictions: []


            }
        },


        methods: {
            async train() {
                this.message = 'Training ...';


                const lossValues = [];
                const accuracyValues = [];

                for (let i = 0; i < TRAIN_BATCHES; i++) {
                    this.batchNumber = i;
                    const batch = mnistData.nextTrainBatch(BATCH_SIZE);

                    let testBatch;
                    let validationData;
                    // Every few batches test the accuracy of the mode.
                    if (i % TEST_ITERATION_FREQUENCY === 0) {
                        testBatch = mnistData.nextTestBatch(TEST_BATCH_SIZE);
                        validationData = [
                            testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
                        ];
                    }

                    // The entire dataset doesn't fit into memory so we call fit repeatedly
                    // with batches.
                    const history = await model.fit(
                        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
                        {batchSize: BATCH_SIZE, validationData, epochs: 1});


                    const loss = history.history.loss[0];
                    const accuracy = history.history.acc[0];


                    // Plot loss / accuracy.
                    this.lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});

                    this.plotLoss()
                    this.lossCreated = true;
//                    ui.plotLosses(lossValues);

                    if (testBatch != null) {
                        this.accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
                        this.plotAccuracy()
                        this.accuracyCreated = true;
//                        ui.plotAccuracies(accuracyValues);
                    }

                    batch.xs.dispose();
                    batch.labels.dispose();
                    if (testBatch != null) {
                        testBatch.xs.dispose();
                        testBatch.labels.dispose();
                    }

                    await tf.nextFrame();
                }
                // this.valuesCreated = true;
                this.message = 'Done!'
            },

            async load() {
                mnistData = new MnistData();
                await mnistData.load()
            },

            async showPredictions() {
                const testExamples = 100;
                const batch = mnistData.nextTestBatch(testExamples);

                tf.tidy(() => {
                    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

                    const axis = 1;
                    const labels = Array.from(batch.labels.argMax(axis).dataSync());
                    const predictions = Array.from(output.argMax(axis).dataSync());

//                    ui.showTestResults(batch, predictions, labels);
                });
            },

            async mnist() {
                await this.load()
                await this.train()
                this.showPredictions()
            },

            plotAccuracy() {
                embed(
                    `#accuracyCanvas`, {
                        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
                        'data': {'values': this.accuracyValues},
                        'mark': {'type': 'line', 'legend': null},
                        'width': 260,
                        'orient': 'vertical',
                        'encoding': {
                            'x': {'field': 'batch', 'type': 'quantitative'},
                            'y': {'field': 'accuracy', 'type': 'quantitative'},
                            'color': {'field': 'set', 'type': 'nominal', 'legend': null},
                        }
                    },
                    {width: 360});

            },


            plotLoss() {

                embed(
                    '#lossCanvas', {
                        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
                        'data': {'values': this.lossValues},
                        'mark': {'type': 'line'},
                        'width': 260,
                        'orient': 'vertical',
                        'encoding': {
                            'x': {'field': 'batch', 'type': 'quantitative'},
                            'y': {'field': 'loss', 'type': 'quantitative'},
                            'color': {'field': 'set', 'type': 'nominal', 'legend': null},
                        }
                    },
                    {width: 360});

            },


            async showPredictions() {

                const testExamples = 100
                const batch = mnistData.nextTestBatch(testExamples);

                tf.tidy(() => {
                    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]))
                    const axis = 1
                    const labels = Array.from(batch.labels.argMax(axis).dataSync())
                    const predictions = Array.from(output.argMax(axis).dataSync())

                    this.showTestResults(batch, predictions, labels)

                });


            },

            showTestResults(batch, predictions, labels) {

                const imagesElement = document.getElementById('images');
                console.log(imagesElement)
                const testExamples = batch.xs.shape[0];
                let totalCorrect = 0;
                for (let i = 0; i < testExamples; i++) {
                    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

                    const div = document.createElement('div');
                    div.className = 'pred-container';

                    const canvas = document.createElement('canvas');
                    canvas.setAttribute('class', 'predictionCanvas');
                    this.draw(image.flatten(), canvas);

                    const pred = document.createElement('div');

                    const prediction = predictions[i];
                    const label = labels[i];
                    const correct = prediction === label;

                    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
                    pred.innerText = `pred: ${prediction}`;

                    div.appendChild(pred);
                    div.appendChild(canvas);

                    imagesElement.appendChild(div);


                }
            },

            draw(image, canvas) {
                const [width, height] = [28, 28];
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                const imageData = new ImageData(width, height);
                const data = image.dataSync();
                for (let i = 0; i < height * width; ++i) {
                    const j = i * 4;
                    imageData.data[j + 0] = data[i] * 255;
                    imageData.data[j + 1] = data[i] * 255;
                    imageData.data[j + 2] = data[i] * 255;
                    imageData.data[j + 3] = 255;
                }
                ctx.putImageData(imageData, 0, 0);
            }


        }
    }


</script>


<style scoped>


</style>