// Declare a route
module.exports = function (fastify, opts, done) {
  const sharp = opts.sharp;

  fastify.post('/create-post-image', async (request, reply) => {
      try {
          const data = request.body;
          let imageBufferArray = data?.imageBufferArray;


          if (imageBufferArray) {
              // Resize and optimize the image
              const optimizedImageBuffer = await sharp(imageBufferArray)
                  .resize({
                      width: 800,
                      height: 1000,
                      fit: sharp.fit.inside,
                      withoutEnlargement: true,
                  })
                  .jpeg({ quality: 80 })
                  .toBuffer();

                

              // Send the optimized image in the response
              reply.send({
                  image: optimizedImageBuffer,
              });
          } else {
              reply.status(400).send({ error: 'Image data is missing.' });
          }
      } catch (error) {
          console.error('Error processing image:', error);
          reply.status(500).send({ error: 'Internal Server Error' });
      }
  });

  done();
};
