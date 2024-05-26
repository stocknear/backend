// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/edit-name-watchlist', async (request, reply) => {
    const data = request.body;
    const watchListId = data?.watchListId;
    const newTitle = data?.title;

    let output;
    
    try {
        await pb.collection("watchlist").update(watchListId, {
            'title': newTitle
        })
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