<template>
    <div>
        <!-- <canvas width="400" height="200"></canvas> -->
        <canvas id="canvas" ref="canvas" width="400" height="400" class="playable-canvas"></canvas>
        <div style="display:none;">
            <img id="source" ref="source" src="../assets/rhino.jpg" width="300" height="227">
        </div>

    </div>
</template>

<script>
    export default {
        mounted() {
//    console.log('you created testing')
//    const img = new Image()
//    const canvas = document.createElement('canvas')
//
//      img.src = '../assets/rhino.jpg'
//    const ctx = canvas.getContext('2d')
//
//        canvas.width = img.width;
//        canvas.height = img.height;
//        ctx.drawImage(img, 0, 0)
//        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
//
            const canvas = this.$refs.canvas;
            const ctx = canvas.getContext('2d')

            const img = this.$refs.source
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.drawImage(img, 33, 71, 104, 124, 21, 20, 87, 104);
//            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const imageData = ctx.getImageData(0, 0, 28, 28);
            console.log(imageData.data.length)


            // console.log(img)
        },

        data() {
            return {
                showImage: false,
                code: 'ctx.drawImage(img, 33, 71, 104, 124, 21, 20, 87, 104);'
            }
        },

        methods: {
            editClicked() {
            },

            resetClicked() {
            },

            drawCanvas() {
                const canvas = this.$refs.canvas;
                const ctx = canvas.getContext('2d')

                const img = this.$refs.source
                ctx.clearRect(0, 0, canvas.width, canvas.height)
                ctx.drawImage(img, 33, 71, 104, 124, 21, 20, 87, 104);

            }
        }


    }
</script>

<style scoped>
    .playable-code {
        border: 1px solid red;
        margin-top: 50px;
        font-size: 20px;
    }

    .playable-button {
        font-size: 20px;
    }

</style>

