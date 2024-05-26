// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/leaderboard', async (request, reply) => {
    const data = request.body;

    const startDate = data?.startDate;
    const endDate = data?.endDate;
    let output;

    try {
        output = await pb.collection("portfolios").getFullList({
            filter: `created >= "${startDate}" && created < "${endDate}"`,
            expand: 'user'
        })

    }
    catch(e) {
        //console.log(e)
        output = []
    }
    

    reply.send({ items: output })

    });

    done();
};