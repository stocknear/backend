// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-user-stats', async (request, reply) => {
    const data = request.body;
    const userId = data?.userId;

    let output;

    try {
        const getNumberOfPosts = await pb.collection("posts").getList(1,1, {
            filter: `user="${userId}"`,
        });
        const numberOfPosts = getNumberOfPosts?.totalItems


        const getNumberOfComments = await pb.collection("comments").getList(1,1, {
            filter: `user="${userId}"`,
        });
        const numberOfComments = getNumberOfComments?.totalItems

        output = {numberOfPosts, numberOfComments}
        console.log(output)

    }
    catch(e) {
        output = {};
    }

    reply.send({ items: output })
    });

    done();
};