// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-portfolio-data', async (request, reply) => {
    const data = request.body;

    const userId = data?.userId
    
    const currentDate = new Date();

	const year = currentDate.getFullYear();
	const month = String(currentDate.getMonth() + 1).padStart(2, '0'); // Month is zero-based
	const day = '01';

	const formattedDate = `${year}-${month}-${day}`; // Output: "yyyy-mm-01"


    //Get Portfolio of user for current month
    const output = await pb.collection("portfolios").getList(1, 500, {
        filter: `user="${userId}" && created >="${formattedDate}" `,
    });
    
    reply.send({ items: output.items })

    });

    done();
};