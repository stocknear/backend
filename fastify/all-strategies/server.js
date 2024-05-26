// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/all-strategies', async (request, reply) => {
    const data = request.body;
    const userId = data?.userId;
    let output;

    try {
        output = await pb.collection("stockscreener").getFullList({
        filter: `user="${userId}"`,
        });
    }
    catch(e) {
        output = [];
    }
    
    reply.send({ items: output })
    });

    done();
};