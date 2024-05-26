// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/update-notifications', async (request, reply) => {
    const data = request.body;

    const notificationList = data?.unreadList;
    let output;

    try {
        const itemsToUpdate = notificationList?.filter(item => !item.readed);
        // Perform updates in parallel
        await Promise.all(itemsToUpdate.map(item =>
            pb.collection("notifications").update(item, { readed: 'true' })
        ));
    
        output = 'success'
    }
    catch(e) {
        output = 'failure';
    }
    
    reply.send({ items: output })
    });

    done();
};