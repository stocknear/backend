// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/feedback', async (request, reply) => {
    const data = request.body;

    try {
        await pb.collection("feedback").create(data)
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