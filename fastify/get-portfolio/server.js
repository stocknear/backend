// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-portfolio', async (request, reply) => {
    const data = request.body;

    const userId = data?.userId;
    // Get the current date
    const currentMonth = new Date();
    const nextMonth = new Date(currentMonth);
    nextMonth.setDate(currentMonth.getDate() + 31); // Add a number of days to ensure next month

    // Set the day to 1 to get the beginning of the current month
    const beginningOfMonth = new Date(currentMonth);
    beginningOfMonth.setDate(1);

    const beginningOfNextMonth = new Date(nextMonth);
    beginningOfNextMonth.setDate(1);

    // Format it as a string if needed
    const startDate = beginningOfMonth.toISOString().split('T')[0];
    const endDate = beginningOfNextMonth.toISOString().split('T')[0];

    //console.log('Start Date:', startDate);
    //console.log('End Date:', endDate);

    const output = await pb.collection("portfolios").getFullList(query_params = {"filter": `user="${userId}" && created >= "${startDate}" && created < "${endDate}"`})
    
    reply.send({ items: output })
    })

    done();
};

