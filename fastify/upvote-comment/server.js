// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/upvote-comment', async (request, reply) => {
    const data = request.body;

    const commentId = data?.commentId;
    const postId = data?.postId;
    const userId = data?.userId;
    console.log(data)
    let output = 'failure';

    try {
    
        
    let doesUserExist = await pb.collection("alreadyVoted").getList(1, 50, {
        filter: `user="${userId}"` && `comment="${commentId}"`,
    })
    
    doesUserExist = doesUserExist?.items?.find(item => item?.user === userId && item?.comment === commentId);

    const votedId = doesUserExist?.id;
    let currentVote = doesUserExist?.type;
    
    //console.log('currentVote: ', currentVote)


    const opPost = await pb.collection('comments').getOne(commentId)

    
    //If user has no history with this post create it
    if( !currentVote || votedId === 'undefined')
    {   
        
        let formDataAlreadyVoted = new FormData();
        formDataAlreadyVoted.append('comment', commentId);
        formDataAlreadyVoted.append('user', userId);
        formDataAlreadyVoted.append('type', 'upvote');
        await pb.collection('alreadyVoted').create(formDataAlreadyVoted);
        
        //create new record for notifications collections
        if (userId !== opPost.user)
        {
            let formDataNotifications = new FormData();
            formDataNotifications.append('opUser', opPost.user);
            formDataNotifications.append('user', userId)
            formDataNotifications.append('comment', commentId);
            formDataNotifications.append('post', postId);
            formDataNotifications.append('notifyType', 'vote');
            
            await pb.collection('notifications').create(formDataNotifications);
            
        }
        
    }

    
    
    if (currentVote === 'downvote')
    {   
        console.log('downvote')
        await pb.collection("comments").update(commentId, {
            "upvote+": 1,
        });
        
        await pb.collection("comments").update(commentId, {
            "downvote-": 1,
        });

        
        await pb.collection("alreadyVoted").update(votedId, {
            "type": 'upvote',
        });

        //Reward: Find user of post and add +1 karma points
        let opPost = await pb.collection('comments').getOne(commentId)
        await pb.collection("users").update(opPost.user, {
            "karma+": 2,
        })
        
        
        
    }
    else if (currentVote === 'neutral' || !currentVote)
    {   

         //Reward: User of post gets +1 karma points
         await pb.collection("users").update(opPost.user, {
            "karma+": 1,
        })

        await pb.collection("comments").update(commentId, {
            "upvote+": 1,
        });
        
        
        await pb.collection("alreadyVoted").update(votedId, {
            "type": 'upvote',
        });

       
        
    
        
    }
    else
    {   

        await pb.collection("comments").update(commentId, {
            "upvote-": 1,
        });
        
        
        await pb.collection("alreadyVoted").update(votedId, {
            "type": 'neutral',
        });

        //Reset Reward: Find user of post and subtract -1 karma points
        await pb.collection("users").update(opPost.user, {
            "karma-": 1,
        })
        
        
    }



    }
    catch(e) {
        console.log(e)
    }
    

    reply.send({ items: output })

    });

    done();
};