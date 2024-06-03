// Optimized postHotness function
function postHotness(upvote, numOfComments, created) {
    const ageInHours = (Date.now() - new Date(created).getTime()) / 36e5; // 36e5 is scientific notation for 3600000
    const hotness = (upvote + numOfComments * 2) / Math.pow(ageInHours + 2, 1.5); // Example calculation
    return hotness;
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
                // In case of sort === 'hot' show the most recent post up to 4 week by ranking them with the function postHotness
                
                let endDate = new Date();
                // Get the date one week earlier
                let startDate = new Date();
                startDate.setDate(endDate.getDate() - 30);

                // Format the dates as needed (e.g., "YYYY-MM-DD")
                let endDateStr = endDate.toISOString().split('T')[0];
                let startDateStr = startDate.toISOString().split('T')[0];

                filter += `&& created >= "${startDateStr}" && created <= "${endDateStr}" && pinned = false`
            }

            posts = (await pb.collection('posts').getList(data?.startPage, 10, {
                sort: sort,
                filter: filter,
                expand: 'user,comments(post),alreadyVoted(post)',
                fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
        
            }))?.items;

            if(data?.sortingPosts === 'hot') {
             // Add hotness property to each post
                posts?.forEach(post => {
                    post.hotness = postHotness(post?.upvote, post?.expand['comments(post)']?.length, post?.created);
                });
                posts?.sort((a, b) => b?.hotness - a?.hotness);
            }

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
                //community page code space
                // In case of sort === 'hot' show the most recent post up to 4 week by ranking them with the function postHotness
                
                if(data?.sortingPosts === 'hot') {

                    let endDate = new Date();
                    // Get the date one week earlier
                    let startDate = new Date();
                    startDate.setDate(endDate.getDate() - 30);

                    // Format the dates as needed (e.g., "YYYY-MM-DD")
                    let endDateStr = endDate.toISOString().split('T')[0];
                    let startDateStr = startDate.toISOString().split('T')[0];

                    filter = `created >= "${startDateStr}" && created <= "${endDateStr}" && pinned = false`
                }
                else {
                    filter = `pinned=false`;
                }
                
                posts = await pb.collection('posts').getList(data?.startPage, 10, {
                    sort: sort,
                    filter: filter,
                    expand: 'user, comments(post), alreadyVoted(post)',
                    fields: "*,expand.user,expand.comments(post), expand.alreadyVoted(post).user,expand.alreadyVoted(post).type"
                    
                });

                posts = posts.items
               // Add hotness property to each post
                posts?.forEach(post => {
                    post.hotness = postHotness(post?.upvote, post?.expand['comments(post)']?.length, post?.created);
                });

                posts?.sort((a, b) => b?.hotness - a?.hotness);
                

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

