// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;
    const serialize = opts.serialize;

    fastify.post('/create-portfolio', async (request, reply) => {
    const data = request.body;


    const formData = {'user': data?.userId, 'accountValue': 100000, 
                    'availableCash': 100000, 'overallReturn': 0, 'rank': 0,
                    'holdings': JSON.stringify([]), 'tradingHistory': '[]',
                    'metrics': JSON.stringify({'alpha': 'n/a', 
                    'beta': 'n/a', 
                    'maxDrawdown': 0
                    })
                }
    let output = 'failure';
    try {
        await pb.collection('portfolios').create(serialize(formData));
        output = 'success';
    }
    catch(e) {
        console.log(e)
    }



    reply.send({ message: output })

    });

    done();
};