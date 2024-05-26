// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/downvote', async (request, reply) => {
    const data = request.body;

    const postId = data?.postId;
    const userId = data?.userId;

    let output = 'failure';


    try {
    
        let doesUserExist = await pb.collection("alreadyVoted").getList(1, 50, {
            filter: `user="${userId}"` && `post="${postId}"`,
        })
    
        doesUserExist = doesUserExist.items.find(item => item.user === userId && item.post === postId);
        //console.log('Does it exist yet: ', doesUserExist)
    
        const votedId = doesUserExist?.id;
        let currentVote = doesUserExist?.type;
    
        const opPost = await pb.collection('posts').getOne(postId)
    
        console.log('currentVote: ', currentVote)
        /*
        console.log('postId: ', postId)
        console.log('votedId: ', votedId)
        console.log('currentVote: ', currentVote)
        console.log('user: ', user);
        */
    
        //If user has no history with this post create it
        if( !currentVote || votedId === 'undefined')
        {   
            console.log('created')
            let formDataAlreadyVoted = new FormData();
            formDataAlreadyVoted.append('post', postId);
            formDataAlreadyVoted.append('user', userId);
            formDataAlreadyVoted.append('notifyType', 'downvote');
            await pb.collection('alreadyVoted').create(formDataAlreadyVoted);
            
        }
    
        
        if (currentVote === 'upvote')
        {
    
            
            await pb.collection("posts").update(postId, {
                "downvote+": 1,
            });
    
            await pb.collection("posts").update(postId, {
                "upvote-": 1,
            });
    
            await pb.collection("alreadyVoted").update(votedId, {
                "type": 'downvote',
            });
    
    
            //if user is the opPost then it should only subtract -1 once
            let opPost = await pb.collection('posts').getOne(postId)
    
            if (opPost.user === userId)
            {
                //Punishment: Find user of post and subtract -1 karma points
                await pb.collection("users").update(opPost.user, {
                    "karma-": 2,
                })
            }
            
            else
            {
                //Punishment: Find user of post and subtract -1 karma points
                await pb.collection("users").update(opPost.user, {
                    "karma-": 2,
                })
    
                //Punishment: User who downvotes post also loose -1 karma points
                await pb.collection("users").update(userId, {
                    "karma-": 2,
                })
            }
            
        }
    
        else if (currentVote === 'neutral' || !currentVote)
        {
    
            if (opPost.user === userId)
            {
                //Punishment: Find user of post and subtract -1 karma points
                await pb.collection("users").update(opPost.user, {
                    "karma-": 1,
                })
            }
            
            else
            {
                //Punishment: Find user of post and subtract -1 karma points
                await pb.collection("users").update(opPost.user, {
                    "karma-": 1,
                })
    
                //Punishment: User who downvotes post also loose -1 karma points
                await pb.collection("users").update(userId, {
                    "karma-": 1,
                })
            }
            
    
    
            await pb.collection("posts").update(postId, {
                "downvote+": 1,
            });
    
            await pb.collection("alreadyVoted").update(votedId, {
                "type": 'downvote',
            });
    
    
    
            
        }
    
        else
        {
            await pb.collection("posts").update(postId, {
                "downvote-": 1,
            });
    
            await pb.collection("alreadyVoted").update(votedId, {
                "type": 'neutral',
            });
    
    
            let opPost = await pb.collection('posts').getOne(postId)
    
            //if user is the opPost then it should only add +1 once
            if (opPost.user === userId)
            {
                //Reset Punishment: Find user of post and add +1 karma points back
                let opPost = await pb.collection('posts').getOne(postId)
                await pb.collection("users").update(opPost.user, {
                    "karma+": 1,
                })
    
            }
            
            else
            {
                //Reset Punishment: Find user of post and add +1 karma points back
                let opPost = await pb.collection('posts').getOne(postId)
                await pb.collection("users").update(opPost.user, {
                    "karma+": 1,
                })
    
                //Reset Punishment: User who removes downvote gets back +1 karma points
                await pb.collection("users").update(userId, {
                    "karma+": 1,
                })
            }
    
    
        }
    

    }
    catch(e) {
        console.log(e)
    }
    

    reply.send({ items: output })

    });

    done();
};