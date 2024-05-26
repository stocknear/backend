// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-user-data', async (request, reply) => {
    const data = request.body;
    const userId = data?.userId
    let output;

    try {
        output = await pb.collection("users").getOne(userId)
    }
    catch(e) {
        output = {};
    }
    
    reply.send({ items: output })
    });

    done();
};