// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/create-watchlist', async (request, reply) => {
    const data = request.body;
    let output;
    
    try {
        output = await pb.collection("watchlist").create(data)
    }
    catch(e) {
        //console.log(e)
        output = [];
    }
    

    reply.send({ items: output })

    });

    done();
};