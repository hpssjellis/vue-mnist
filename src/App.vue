<template>
  <div id="app">
    <div class="container">
      <HelloWorld/>
      <Main></Main>
    </div>
    <!--<Testing></Testing>-->
  </div>
</template>

<script>
import HelloWorld from './components/HelloWorld'
import Main from './components/Main'
import Testing from './components/Testing'

export default {
  name: 'App',
  components: {
     HelloWorld,
       Main,
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
  margin-top: 60px;
}

.container {
  padding-left: 20px;
  padding-right: 20px;
}

.vega-actions {
  display: none;
}
</style>
