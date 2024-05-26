// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/delete-strategy', async (request, reply) => {
    const data = request.body;
    let output;

    try {
        await pb.collection("stockscreener").delete(data['strategyId'])
        output = 'success';
    }
    catch(e) {
        output = 'failure';
    }
    
    reply.send({ items: output })
    });

    done();
};