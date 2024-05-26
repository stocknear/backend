
function processHoldingsList(holdings, soldTicker) {
    const stockGroups = {};
  
    for (const stock of holdings) {
      if (stock.symbol in stockGroups) {
        stockGroups[stock.symbol].totalCost -= stock.boughtPrice * stock.numberOfShares;
        stockGroups[stock.symbol].totalShares -= stock.numberOfShares;
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
  
      if (stock.symbol === soldTicker) {
        // Only update dailyChange and sinceBoughtChange for the sold ticker
        stockGroups[stock.symbol].dailyChange = stock.dailyChange;
        stockGroups[stock.symbol].sinceBoughtChange = stock.sinceBoughtChange;
        stockGroups[stock.symbol].currentPrice = stock.currentPrice;
      }
    }
  
    const updatedHoldings = [];
  
    for (const symbol in stockGroups) {
      const { totalCost, totalShares, name, assetType, currentPrice, dailyChange, sinceBoughtChange } = stockGroups[symbol];
      let finalBoughtPrice;
  
      if (totalShares !== 0) {
        finalBoughtPrice = totalCost / totalShares;
      } else {
        // If totalShares is 0, set finalBoughtPrice to 0
        finalBoughtPrice = 0;
      }
  
      const updatedStock = {
        symbol,
        name,
        assetType,
        boughtPrice: finalBoughtPrice,
        currentPrice,
        dailyChange,
        sinceBoughtChange: Number(((currentPrice / finalBoughtPrice - 1) * 100)?.toFixed(2)),
        numberOfShares: totalShares,
      };
      updatedHoldings.push(updatedStock);
    }
  
    return updatedHoldings;
  }
  
// Declare a route
module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;
    fastify.post('/sell-stock', async (request, reply) => {

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



    const { userId, symbol, name, numberOfShares, soldPrice } = data;

    let currentPortfolio = await pb.collection("portfolios").getList(1, 500, {
        filter: `user="${userId}" && created >="${formattedDate}" `,
    });

    currentPortfolio = currentPortfolio?.items[0];

    let holdings = currentPortfolio?.holdings || [];
    let tradingHistory = currentPortfolio?.tradingHistory || [];
    let availableCash = currentPortfolio?.availableCash || 0;

    // Find the stock in the holdings list
    const stockIndex = holdings.findIndex((stock) => stock.symbol === symbol);

    if(stockIndex !== -1)
    {
        const soldValue = numberOfShares * soldPrice;
        // Reduce the number of shares from the existing holding
        holdings[stockIndex].numberOfShares -= numberOfShares;

        if (holdings[stockIndex].numberOfShares <= 0) {
            // If all shares are sold, remove the stock from the holdings list
            holdings.splice(stockIndex, 1);
        }

        // Add the sold value to the available cash
        availableCash += soldValue;

        // Recalculate the updated holdings list
        const updatedHoldings = processHoldingsList(holdings);

        tradingHistory.push({'symbol': symbol,
                      'name': name,
                      'assetType': data['assetType'],
                      'numberOfShares': numberOfShares,
                      'price': Number(soldPrice),
                      'type': 'sell',
                      'date': new Date()});

        try {
			
            await pb.collection("portfolios").update(currentPortfolio?.id, {
                "holdings": updatedHoldings,
                "availableCash": availableCash,
                "tradingHistory": tradingHistory,
            })
            output = 'success';
    
        } catch (err) {
            output = 'failure';
        }

    }

    }
    else {
      output = 'marketClosed'
    }

    reply.send({ message: output })
    })

    done();
};

