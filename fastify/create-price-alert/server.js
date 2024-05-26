// Declare a route


module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/create-price-alert', async (request, reply) => {

    const data = request.body;
    
    let output;

    let newAlert = {
        'user': data['userId'],
        'symbol': data['symbol']?.toUpperCase(),
        'name': data['name'],
        'assetType': data['assetType']?.toLowerCase(),
        'targetPrice': Number(data['targetPrice']),
        'condition': data['condition']?.toLowerCase(),
        'priceWhenCreated': Number(data['priceWhenCreated']),
        'triggered': false,
    }

    
    try {

        await pb.collection("priceAlert")?.create(newAlert)
        output = 'success';

      } catch (err) {
          output = 'failure'
      }


    reply.send({ items: output })
    })

    done();
};

