// Declare a route


function processHoldingsList(holdings) {
    const stockGroups = {};
  
    for (const stock of holdings) {
      if (stock.symbol in stockGroups) {
        stockGroups[stock.symbol].totalCost += stock.boughtPrice * stock.numberOfShares;
        stockGroups[stock.symbol].totalShares += stock.numberOfShares;
      } else {
        stockGroups[stock.symbol] = {
          totalCost: stock.boughtPrice * stock.numberOfShares,
          totalShares: stock.numberOfShares,
          name: stock.name,
          assetType: stock.assetType,
          currentPrice: stock.currentPrice,
          dailyChange: stock.dailyChange,
          sinceBoughtChange: stock.sinceBoughtChange,
        };
      }
  
      // Update dailyChange automatically
      stockGroups[stock.symbol].dailyChange = stock.dailyChange;
      stockGroups[stock.symbol].sinceBoughtChange = stock.sinceBoughtChange;
      stockGroups[stock.symbol].currentPrice = stock.currentPrice;
    }
  
    const updatedHoldings = [];
  
    for (const symbol in stockGroups) {
      const { totalCost, totalShares, name, assetType, currentPrice, dailyChange, sinceBoughtChange } = stockGroups[symbol];
      const finalBoughtPrice = totalCost / totalShares;
      const updatedStock = {
        symbol,
        name,
        assetType,
        boughtPrice: finalBoughtPrice,
        currentPrice,
        dailyChange,
        sinceBoughtChange: Number(((currentPrice/finalBoughtPrice -1) * 100)?.toFixed(2)),
        numberOfShares: totalShares,
      };
      updatedHoldings.push(updatedStock);
    }
  
    return updatedHoldings;
  }


module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/buy-stock', async (request, reply) => {

    const holidays = ['2024-01-01', '2024-01-15','2024-02-19','2024-03-29','2024-05-27','2024-06-19','2024-07-04','2024-09-02','2024-11-28','2024-12-25'];
    const currentDate = new Date().toISOString().split('T')[0];

    // Get the current time in the ET time zone
    const etTimeZone = 'America/New_York';
    const currentTime = new Date().toLocaleString('en-US', { timeZone: etTimeZone });

    // Determine if the NYSE is currently open or closed
    const currentHour = new Date(currentTime).getHours();
    const isWeekend = new Date(currentTime).getDay() === 6 || new Date(currentTime).getDay() === 0;
    const isBeforeMarketOpen = currentHour < 9 || (currentHour === 9 && new Date(currentTime).getMinutes() < 30);
    const isAfterMarketClose = currentHour >= 16;

    const isStockMarketOpen = !(isWeekend || isBeforeMarketOpen || isAfterMarketClose || holidays?.includes(currentDate));
    let output;
    if (isStockMarketOpen === true) {
      
      const data = request.body;
      const currentDate = new Date();
      const year = currentDate.getFullYear();
      const month = String(currentDate.getMonth() + 1).padStart(2, '0'); // Month is zero-based
      const day = '01';

      const formattedDate = `${year}-${month}-${day}`; // Output: "yyyy-mm-01"

      const userId = data?.userId;

      let newHolding = {'symbol': data['symbol'],
                        'name': data['name'],
                        'assetType': data['assetType'],
                        'numberOfShares': data['numberOfShares'],
                        'boughtPrice': data['boughtPrice'],
                        'currentPrice': data['boughtPrice'],
                        'dailyChange': 0,
                        'sinceBoughtChange': 0 }

      let currentPortfolio = await pb.collection("portfolios").getList(1, 500, {
          filter: `user="${userId}" && created >="${formattedDate}" `,
      });

      currentPortfolio = currentPortfolio?.items[0];

      let holdings = currentPortfolio?.holdings || [];

      let tradingHistory = currentPortfolio?.tradingHistory || [];

      let availableCash = currentPortfolio?.availableCash - data['estimatedTotal'];

      holdings.push(newHolding)

      const updatedHoldings = processHoldingsList(holdings)


      tradingHistory.push({'symbol': data['symbol'],
                          'name': data['name'],
                          'assetType': data['assetType'],
                          'numberOfShares': data['numberOfShares'],
                          'price': Number(data['boughtPrice']),
                          'type': 'buy',
                          'date': new Date()});

      try {

          await pb.collection("portfolios").update(currentPortfolio?.id, {
              "holdings": updatedHoldings,
              "availableCash": availableCash,
              "tradingHistory": tradingHistory,
          })
          output = 'success';

      } catch (err) {
          output = 'failure'
      }
    }
    else {
      output = 'marketClosed'
    }

    reply.send({ items: output })
    })

    done();
};

