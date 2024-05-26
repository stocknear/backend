// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/delete-price-alert', async (request, reply) => {
    const data = request.body;
    const priceAlertIdList = data?.priceAlertIdList;
    let output;
    try {
        for (const item of priceAlertIdList) {
            await pb.collection("priceAlert")?.delete(item)
        }
        output = 'success';
    }
    catch(e) {
        //console.log(e)
        output = 'failure';
    }
    
    reply.send({ items: output })

    });

    done();
};