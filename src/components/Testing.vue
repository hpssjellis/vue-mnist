<template>
  <div>
    <!-- <canvas width="400" height="200"></canvas> -->
    <div v-if="showImage">
      <p>You are in the image div</p> 
      <img src="https://mdn.mozillademos.org/files/5397/rhino.jpg" alt="" width="300" height="227">
    </div>

    <div class="playable-buttons">
      <input type="button" value="Edit" @click="editClicked">
      <input type="button" value="Reset" @click="resetClicked">
    </div>
    <textarea class="playable-code" v-model="code" rows="4" cols="50"></textarea>

    </div>
</template>

<script>
export default {
  created(){
    console.log('you created testing')
    const img = new Image()
    const canvas = document.createElement('canvas')
    
      img.src = '../assets/rhino.jpg'
    const ctx = canvas.getContext('2d')
    
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      
    


    // console.log(img)
  },
  
  data(){
    return {
      showImage: false,
      code: 'ctx.drawImage(img, 33, 71, 104, 124, 21, 20, 87, 104);'
    }
  },

  methods: {
    editClicked(){
      console.log('edit clicked')
    },

    resetClicked(){
      console.log('reset button was clicked')
    }
  }

  
}
</script>

<style scoped>
.playable-code{
  border: 1px solid red;
  margin-top: 50px;
  font-size: 20px;
}

.playable-button{
  font-size: 20px;
}

</style>

