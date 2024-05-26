// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-notifications', async (request, reply) => {
    const data = request.body;

    const userId = data?.userId;
    let output;

    try {
        output = await pb.collection("notifications").getFullList({
            filter: `opUser="${userId}"`,
            expand: 'user,post,comment',
            sort: '-created'
        });

    }
    catch(e) {
        //console.log(e)
        output = []
    }
    
    reply.send({ items: output })

    });

    done();
};