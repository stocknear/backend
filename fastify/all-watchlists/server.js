// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/all-watchlists', async (request, reply) => {
    const data = request.body;
    const userId = data?.userId;

    let output;
    
    try {
        output = await pb.collection("watchlist").getFullList({
            filter: `user="${userId}"`
        })
    }
    catch(e) {
        //console.log(e)
        output = {};
    }

    reply.send({ items: output })

    });

    done();
};