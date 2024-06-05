// Optimized postHotness function
function postHotness(upvotes, created) {
    let s = 0;
    for (let i = 1; i <= upvotes; i++) {
        if (i <= 3) {
            s += 1;
        } else if (i <= 6) {
            s += 3;
        } else if (i <= 10) {
            s += 3;
        } else if (i <= 20) {
            s += 4;
        } else if (i <= 40) {
            s += 5;
        } else {
            s += 6;
        }
    }

    const order = Math.log10(Math.max(Math.abs(s), 1));
    let sign = 0;
    if (s > 0) {
        sign = 1;
    } else if (s < 0) {
        sign = -1;
    }

    const interval = 45000; // or 69000
    const createdDate = new Date(created);
    const seconds = (createdDate.getTime() / 1000);
    const hotness = order + (sign * seconds) / interval;
    return Math.round(hotness * 10000000);
}


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
            
            if(data?.sortingPosts === 'hot') {
                //community page code space
                // In case of sort === 'hot' show the most recent post up to 7 week by ranking them with the function postHotness
                
                let endDate = new Date();
                // Get the date one week earlier
                let startDate = new Date();
                startDate.setDate(endDate.getDate() - 30);
                endDate.setDate(endDate.getDate() + 1)

                // Format the dates as needed (e.g., "YYYY-MM-DD")
                let endDateStr = endDate.toISOString().split('T')[0];
                let startDateStr = startDate.toISOString().split('T')[0];

                filter += `&& created >= "${startDateStr}" && created <= "${endDateStr}" && pinned = false`
            }

            posts = (await pb.collection('posts').getList(data?.startPage, 5, {
                sort: '-created',
                filter: filter,
                expand: 'user,comments(post),alreadyVoted(post)',
                fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
        
            }))?.items;

            if(data?.sortingPosts === 'hot') {
             // Add hotness property to each post
                posts?.forEach(post => {
                    post.hotness = postHotness(post?.upvote, post?.created);
                });
                posts?.sort((a, b) => b?.hotness - a?.hotness);
            }

        }
        else {

            if (data?.userId) {
            
            posts = (await pb.collection('posts').getList(data?.startPage, 5, {
                sort: sort,
                filter: `user="${data?.userId}" && pinned=false`,
                expand: `user,comments(post),alreadyVoted(post)`,
                fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
            }))?.items;


            }

            else if (data?.filterTicker) {
            
                posts = await pb.collection('posts').getList(data?.startPage, 5, {
                    sort: sort,
                    filter: `tagline="${data?.filterTicker}" && pinned=false`,
                    expand: `user,comments(post),alreadyVoted(post)`,
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                }) ;
        
                }

            else {
                //community page code space
                // In case of sort === 'hot' show the most recent post up to 7 week by ranking them with the function postHotness
                
                if(data?.sortingPosts === 'hot') {

                    let endDate = new Date();
                    // Get the date one week earlier
                    let startDate = new Date();
                    startDate.setDate(endDate.getDate() - 30);
                    endDate.setDate(endDate.getDate() + 1)

                    // Format the dates as needed (e.g., "YYYY-MM-DD")
                    let endDateStr = endDate.toISOString().split('T')[0];
                    let startDateStr = startDate.toISOString().split('T')[0];

                    filter = `created >= "${startDateStr}" && created <= "${endDateStr}" && pinned = false`
                }
                else {
                    filter = `pinned=false`;
                }
                posts = await pb.collection('posts').getList(1, 5, {
                    sort: '-created',
                    filter: filter,
                    expand: 'user, comments(post), alreadyVoted(post)',
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                    
                });

                posts = posts.items
                // Add hotness property to each post
                if(data?.sortingPosts === 'hot') {
                    posts?.forEach(post => {
                        post.hotness = postHotness(post?.upvote, post?.created);
                    });

                    posts?.sort((a, b) => b?.hotness - a?.hotness);
                }

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

