// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-one-post', async (request, reply) => {
    const data = request.body;

    const postId = data?.postId;

    const output = await pb.collection('posts').getOne(postId, {
        expand: 'user,alreadyVoted(post)',
        fields: "*,expand.user,expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
        
    });

    reply.send({ items: output })
    })

    done();
};

