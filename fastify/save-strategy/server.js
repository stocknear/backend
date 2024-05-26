// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/save-strategy', async (request, reply) => {
    const data = request.body;
    let output;

    try {
        output = await pb.collection("stockscreener").update(data['strategyId'], {
            'rules': data['rules']
        })
    }
    catch(e) {
        output = {};
    }
    
    reply.send({ items: output })
    });

    done();
};