// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/delete-post', async (request, reply) => {
    const data = request.body;
    const postId = data?.postId;
    const userId = data?.userId;
    const moderator = 'db5s41oatgoeh0q' //moderators can always delete post
    let output;

    try {
        if(moderator === userId)
        {
            await pb.collection('posts').delete(postId);
            output = 'success';
        }
        else {
            const res = await pb.collection('posts').getOne(postId);
            if (res?.user === userId)
            {
                await pb.collection('posts').delete(postId);
                output = 'success';
            }
            else {
                output = 'failure';
            }
        }
    }
    catch(e) {
        console.log(e)
        output = 'failure';
    }
    
    reply.send({ items: output })
    });

    done();
};