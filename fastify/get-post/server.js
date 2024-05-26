// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-post', async (request, reply) => {
    const data = request.body;

    let filter;
    const sort = data?.sortingPosts === 'hot' ? '-upvote' : '-created';
    let pinnedPost;
    let posts;

    try {

        
        if (data?.seenPostId.length !==0)
        {
            filter = data?.seenPostId?.map((id) => `id!="${id}"`).join("&&");

            //applies only for profile and user directory
            if (data?.userId) {
                filter += `&& user="${data?.userId}" && pinned=false`;
            }

            if (data?.filterTicker) {
                filter += `&& tagline="${data?.filterTicker}" && pinned=false`;
            }

            posts = (await pb.collection('posts').getList(data?.startPage, 10, {
                sort: sort,
                filter: filter,
                expand: 'user,comments(post),alreadyVoted(post)',
                fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
        
            }))?.items;
        }
        else {

            if (data?.userId) {
            
            posts = (await pb.collection('posts').getList(data?.startPage, 10, {
                sort: sort,
                filter: `user="${data?.userId}" && pinned=false`,
                expand: `user,comments(post),alreadyVoted(post)`,
                fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
            }))?.items;


            }

            else if (data?.filterTicker) {
            
                posts = await pb.collection('posts').getList(data?.startPage, 10, {
                    sort: sort,
                    filter: `tagline="${data?.filterTicker}" && pinned=false`,
                    expand: `user,comments(post),alreadyVoted(post)`,
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                }) ;
        
                }

            else {
                posts = await pb.collection('posts').getList(data?.startPage, 10, {
                    sort: sort,
                    filter: `pinned=false`,
                    expand: 'user, comments(post), alreadyVoted(post)',
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                    
                });

                posts = posts.items

                pinnedPost = await pb.collection('posts').getFullList({
                    filter: `pinned=true`,
                    sort: '-created',
                    expand: `user,comments(post),alreadyVoted(post)`,
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                });
                
                for (let i = pinnedPost?.length - 1; i >= 0; i--) {
                    posts?.unshift(pinnedPost[i]);
                }

            }

            }

    }
    catch(e)
    {
        //console.log(e)
        posts = [];
    }


    reply.send({ items: posts })
    })

    done();
};

