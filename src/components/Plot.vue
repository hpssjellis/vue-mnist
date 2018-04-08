<template>
    <div class="canvases">
        <div class="label" id="accuracy-label"></div>
        <div :id="id"></div>
    </div>
</template>

<script>
    import embed from 'vega-embed'


    export default {

        props: ['id', 'lossValues', 'accuracyValues', 'accuracyOrLoss'],

        data() {
            return {}
        },

        mounted() {
            console.log('printing accuracy values')
            embed(
                `#${this.id}`, {
                    '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
                    'data': {'values': this.lossValues },
                    'mark': {'type': 'line', 'legend': null},
                    'width': 260,
                    'orient': 'vertical',
                    'encoding': {
                        'x': {'field': 'batch', 'type': 'quantitative'},
                        'y': {'field': 'loss', 'type': 'quantitative'},
                        'color': {'field': 'set', 'type': 'nominal', 'legend': null},
                    }
                },
                {width: 360});
        }
    }
</script>

<style scoped>
    .canvases {
        display: inline-block;
        width: 460px;
    }

</style>

