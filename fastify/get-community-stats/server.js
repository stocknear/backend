module.exports = function (fastify, opts, done) {
    const pb = opts.pb;

    fastify.get('/get-community-stats', async (request, reply) => {
        let output;
        let totalUsers = 0;
        let totalPosts = 0;
        let totalComments = 0;

        try {
            totalUsers = (await pb.collection("users").getList(1, 1))?.totalItems;
            totalPosts = (await pb.collection("posts").getList(1, 1))?.totalItems;
            totalComments = (await pb.collection("comments").getList(1, 1))?.totalItems;

          
            output = { totalUsers, totalPosts, totalComments };

        } catch (e) {
            console.error(e);
            output = { totalUsers, totalPosts, totalComments };
        }

        reply.send({ items: output });
    });

    done();
};
