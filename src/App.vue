<template>
    <div id="app">
            <Hero/>
        <div class="container">
            <HelloWorld/>
            <Main></Main>
        </div>
        <!--<Testing></Testing>-->
    </div>
</template>

<script>
    require('../node_modules/bulma/css/bulma.css')
    import Main from './components/Main'
    import Hero from './components/Hero'
    import Testing from './components/Testing'


    export default {
        name: 'App',
        components: {
            Main,
            Hero
//      Testing
        }
    }
</script>

<style>
    #app {
        font-family: 'Avenir', Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        /*text-align: center;*/
        color: #2c3e50;
    }

    .vega-actions {
        display: none;
    }


</style>
