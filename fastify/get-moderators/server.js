// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.get('/get-moderators', async (request, reply) => {
    let output;

    try {
        output = await pb.collection("moderators").getFullList({
            expand: 'user'
        })
    }
    catch(e) {
        output = [];
    }
    
    reply.send({ items: output })
    });

    done();
};