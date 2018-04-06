<template>
    <div class="container">

        <div id="message">{{ message }}</div>

        <div id="stats">
            <div class="canvases">
                <div class="label" id="loss-label"></div>
                <div id="lossCanvas"></div>
                <plot
                        v-bind:id="lossCanvas"
                        v-bind:lossValues="lossValues"
                        v-if="valuesCreated"
                >
                </plot>
                <plot>
                    v-bind:id="accuracyCanvas"
                    v-bind:accuracyValues="accuracyValues"
                    v-if="accuracyCreated"
                </plot>
            </div>
            <div class="canvases">
                <div class="label" id="accuracy-label"></div>
                <div id="accuracyCanvas"></div>
            </div>
        </div>
        <div id="images"></div>
    </div>
</template>


<script>
    import * as tf from '@tensorflow/tfjs'
    import { MnistData } from '../data.js'

    import Plot from '../components/Plot.vue'

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

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2]}))
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelIniliazer: 'varianceScaling'
    }))
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2]}))
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

        data() {
            return {
                message: 'Loading data ...',
                lossCanvas: 'lossCanvas',
                lossValues: [],
                accuracyValues: [],
                valuesCreated: false

            }
        },

        components: {Plot},

        methods: {
            async train() {
                this.message = 'Training ...';


                const lossValues = [];
                const accuracyValues = [];

                for (let i = 0; i < TRAIN_BATCHES; i++) {
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
//                    ui.plotLosses(lossValues);

                    if (testBatch != null) {
                        this.accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
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
                this.valuesCreated = true;
                console.log('you finished train')
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
            }

        }
    }




</script>


<style>

</style>