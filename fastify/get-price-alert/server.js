

module.exports = function (fastify, opts, done) {
    const pb = opts.pb;
    const fs = opts.fs
    const path = opts.path

    fastify.post('/get-price-alert', async (request, reply) => {
        const data = request.body;
        const userId = data?.userId;

        let output;

        try {
            output = await pb.collection("priceAlert").getFullList({
                filter: `user="${userId}" && triggered=false`
            });

            // Read the JSON file for each symbol in the output list
            const itemsWithQuotes = await Promise.all(output.map(async (item) => {
                const symbol = item.symbol;
                try {
                    const filePath = path.join(__dirname, `../../app/json/quote/${symbol}.json`);
                    const fileData = fs.readFileSync(filePath, 'utf8');
                    const jsonData = JSON.parse(fileData);
                    
                    // Extract only the desired fields from the JSON data
                    const { changesPercentage, price, volume } = jsonData;
                    

                    
                    return { ...item, changesPercentage, price, volume};
                    
                } catch (error) {
                    // Handle errors if file reading or parsing fails
                    console.error(`Error reading or parsing JSON for symbol ${symbol}: ${error}`);
                    return item;
                }
            }));
            reply.send({ items: itemsWithQuotes });
        } catch (e) {
            console.error(e);
            reply.send({ items: [] });
            //reply.status(500).send({ error: "Internal Server Error" });
        }
    });

    done();
};
