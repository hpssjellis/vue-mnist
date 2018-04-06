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
                    const batch = data.nextTrainBatch(BATCH_SIZE)

                    let testBatch
                    let validationData

                    if (i % TEST_ITERATION_FREQUENCY === 0) {
                        testBatch = data.nextTestBatch(TEST_BATCH_SIZE)
                        validationData = [
                            testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1])
                        ]
                    }

                    const history = await model.fit(
                        batch.xs.reshape([BATCH_SIZE, 28, 28, 1], batch.labels,
                            {batchSize: BATCH_SIZE, validationData, epochs: 1}
                        )
                    );

                    const loss = history.history.loss[0];
                    const accuracy = history.history.acc[0]

                    this.lossValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'})

                }
                this.valuesCreated = true;
            },

            async load() {
                const mnistData = new MnistData();
                await mnistData.load()
            },

            async mnist() {
                await this.load()
                await this.train()
                showPredictions()
            }

        }
    }




</script>


<style>

</style>