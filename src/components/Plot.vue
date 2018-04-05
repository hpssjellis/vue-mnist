<template>
    <div :id="id"></div>
</template>

<script>
    import embed from 'vega-embed'


    export default{

        props: [ 'id', 'lossValues'],

        data() {
            return {
            }
        },

        created() {
                embed(
                    `#${this.id}`, {
                        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
                        'data': {'values': this.lossValues },
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
        }
    }
</script>

<style>

</style>

