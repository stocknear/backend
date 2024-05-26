// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;
    const serialize = opts.serialize;

    fastify.post('/update-watchlist', async (request, reply) => {
    const data = request.body;

    const userId = data?.userId;
    const ticker = data?.ticker;
    const watchListId = data?.watchListId;
    let output;

    try {
        const watchList = await pb.collection("watchlist").getOne(watchListId);

        if (watchList?.ticker?.includes(ticker)) {
        // Remove ticker from the watchlist.
        const newTickerList = watchList?.ticker.filter(item => item !== ticker);
        output = await pb.collection("watchlist").update(watchListId, { ticker: newTickerList });
        } else {
        // Add ticker to the watchlist.
        const newTickerList = [...watchList?.ticker, ticker];
        output = await pb.collection("watchlist").update(watchListId, { ticker: newTickerList });
        }

    }
    catch(e) {
        //console.log(e)
        output = await pb.collection("watchlist").create(serialize({'user': userId, 'ticker': JSON.stringify([ticker]), 'title': 'Favorites' }));
    }
    

    reply.send({ items: output })

    });

    done();
};