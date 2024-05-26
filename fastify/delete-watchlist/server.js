// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/delete-watchlist', async (request, reply) => {
    const data = request.body;
    const watchListId = data?.watchListId
    let output;

    try {
        await pb.collection("watchlist").delete(watchListId)
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