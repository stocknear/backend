// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-strategy', async (request, reply) => {
    const data = request.body;
    let output;

    try {
        output = await pb.collection("stockscreener").getOne(data['strategyId'])
    }
    catch(e) {
        output = {};
    }
    
    reply.send({ items: output })
    });

    done();
};