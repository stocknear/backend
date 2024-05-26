// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/delete-comment', async (request, reply) => {
    const data = request.body;

    let output;
	const userId = data?.userId;
	const commentUserId = data?.commentUser;
	const commentId = data.commentId

	//Each delete gives the user -1 Karma points

	let checkModerator = await pb.collection('moderators').getList(1, 50)

    //OP and moderators have the right to delete comments
	if (commentUserId === userId || checkModerator.items.some((item) => item.user === userId))
	{

		try {
			
			await pb.collection('comments').delete(commentId);
            await pb.collection("users").update(commentUserId, {
                "karma-": 1,
            })

			output = 'success';
            

		} catch (err) {
			output = 'failure'
            console.log(err)
		}
	}
	else {
		output = 'failure';
	}

    

    reply.send({ message: output })

    });

    done();
};