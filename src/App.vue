<template>
  <div id="app">
    <img src="./assets/logo.png">
     <HelloWorld/>
     <Main></Main>
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
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
