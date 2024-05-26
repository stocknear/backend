// Declare a route
module.exports = function (fastify, opts, done) {
    const got = opts.got;
    const cheerio = opts.cheerio;
    const sharp = opts.sharp;
  
fastify.post('/create-post-link', async (request, reply) => {
    const data = request.body;
    const url = data?.link;
    let description;
    let imageBuffer;

    try {
        const response = await got(url, {
          headers: {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
          },
          responseType: 'buffer',
        });
  
        const $ = cheerio.load(response.body);
  
        description = $('head meta[property="og:description"]').attr('content');
        let image = $('head meta[property="og:image"]').attr('content');
  
        if (!image) {
          let largestSize = 0;
          let largestImage = '';
  
          $('img').each(async function () {
            if ($(this).attr('src') && $(this).attr('src').match(/\.(webp|jpg|jpeg|png|gif)$/)) {
              try {
                imageBuffer = await got($(this).attr('src'), {
                  headers: {
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                  },
                  responseType: 'buffer',
                }).then((response) => response.body);
          
                const metadata = await sharp(imageBuffer).metadata();
                const imageSize = metadata.width * metadata.height;
          
                if (imageSize > largestSize) {
                  largestSize = imageSize;
                  largestImage = $(this).attr('src');
                }
              } catch (error) {
                console.error('Error getting image:', error);
              }
            }
          });
  
          image = largestImage;
        }
  
        imageBuffer = await got(image, {
          headers: {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
          },
          responseType: 'buffer',
        }).then((response) => response.body);
  
      } catch (e) {
        console.error(e);
      }
  
      // Check if imageBlob is not null before sending it in the response
    reply.send({
          description: description,
          image: imageBuffer,
        })

    });
  
    done();
  };
  