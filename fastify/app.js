let serverRunning = false;

const fastify = require("fastify")({});
const cors = require("@fastify/cors");

//Load API KEYS
require("dotenv").config({ path: "../app/.env" });
const fmpAPIKey = process.env.FMP_API_KEY;
//const mixpanelAPIKey =  process.env.MIXPANEL_API_KEY;

//const Mixpanel = require('mixpanel');
//const UAParser = require('ua-parser-js');

const fs = require("fs");
const path = require("path");
//const pino = require('pino');

//const mixpanel = Mixpanel.init(mixpanelAPIKey, { debug: false });

//const PocketBase = require("pocketbase/cjs");
//const pb = new PocketBase("http://127.0.0.1:8090");

// globally disable auto cancellation
//See https://github.com/pocketbase/js-sdk#auto-cancellation
//Bug happens that get-post gives an error of auto-cancellation. Hence set it to false;
//pb.autoCancellation(false);

const { serialize } = require("object-to-formdata");

// Register the CORS plugin
//Add Cors so that only localhost and my stocknear.com can send acceptable requests
fastify.register(cors);
const corsMiddleware = (request, reply, done) => {
  const allowedOrigins = [
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://stocknear.com",
    "https://www.stocknear.com",
    "http://stocknear.com",
    "http://www.stocknear.com",
  ];

  const origin = request?.headers?.origin;
  if (!origin || allowedOrigins?.includes(origin)) {
    reply.header("Access-Control-Allow-Origin", origin || "*");
    reply.header("Access-Control-Allow-Methods", "GET,POST");
    reply.header("Access-Control-Allow-Headers", "Content-Type");
    done();
  } else {
    reply.code(403).send({ error: "Forbidden" });
  }
};
fastify.addHook("onRequest", corsMiddleware);


function wait(ms) {
  var start = new Date().getTime();
  var end = start;
  while (end < start + ms) {
    end = new Date().getTime();
  }
}

fastify.register(require("@fastify/websocket"));

const WebSocket = require("ws");

let isSend = false;
let sendInterval;

function formatTimestampNewYork(timestamp) {
  const d = new Date(timestamp / 1e6);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
    .format(d)
    .replace(/(\d+)\/(\d+)\/(\d+),/, "$3-$1-$2")
    .replace(",", "");
}


fastify.register(async function (fastify) {
  fastify.get(
    "/realtime-crypto-data",
    { websocket: true },
    (connection, req) => {
      // Send a welcome message to the client

      // Listen for incoming messages from the client
      connection.socket.on("message", (message) => {
        symbol = message.toString("utf-8");
        console.log("Received message from client:", symbol);

        // If you want to dynamically update the subscription based on client's message
        updateSubscription();
      });

      //======================
      const login = {
        event: "login",
        data: {
          apiKey: fmpAPIKey,
        },
      };

      const subscribe = {
        event: "subscribe",
        data: {
          ticker: "", // Initial value; will be updated dynamically
        },
      };

      function updateSubscription() {
        subscribe.data.ticker = symbol;
      }

      // Create a new WebSocket instance for your backend
      const ws = new WebSocket("wss://crypto.financialmodelingprep.com");

      // Handle WebSocket connection open

      ws.on("open", function open() {
        ws.send(JSON.stringify(login));
        wait(2000); //2 seconds in milliseconds
        ws.send(JSON.stringify(subscribe));
      });

      // Handle WebSocket errors
      ws.on("error", function (error) {
        console.error("WebSocket error:", error);
        // Handle the error gracefully, you might want to notify the client or log it.
        // For now, let's close the connection if an error occurs
        connection.socket.close();
      });

      // Handle WebSocket messages
      ws.on("message", function (data, flags) {
        const stringData = data.toString("utf-8");

        if (connection.socket.readyState === WebSocket.OPEN && !isSend) {
          connection.socket.send(stringData);
          //console.log(stringData)
          isSend = true;
          setTimeout(() => {
            isSend = false;
          }, 2000);

          //wait(2000);
        }
        //wait(2000);
      });

      //======================

      // Handle client disconnect
      connection.socket.on("close", () => {
        console.log("Client disconnected");
        connection?.socket?.close();
        // Check if the WebSocket is open before trying to close it
        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.close();
          } catch (e) {
            console.error("Error while closing WebSocket:", e);
          }
        }
      });
    }
  );
});



fastify.register(async function (fastify) {
  fastify.get("/options-flow-reader", { websocket: true }, (connection, req) => {
    let sendInterval;
    let pingInterval;
    let sendTimeout = null;
    let lastSentData = [];

    const sendData = async () => {
      if (sendTimeout) {
        clearTimeout(sendTimeout);
        sendTimeout = null;
      }

      const filePath = path.join(__dirname, "../app/json/options-flow/feed/data.json");
      
      try {
        if (fs.existsSync(filePath)) {
          const fileData = fs.readFileSync(filePath, "utf8").trim();

          if (!fileData) {
            sendTimeout = setTimeout(sendData, 2000);
            return;
          }

          let parsedData;
          try {
            parsedData = JSON.parse(fileData);
          } catch (jsonErr) {
            sendTimeout = setTimeout(sendData, 2000);
            return;
          }

          if (parsedData.length > lastSentData.length && connection.socket.readyState === 1) {
            connection.socket.send(JSON.stringify(parsedData));
            lastSentData = parsedData;
          }
        } else {
          sendTimeout = setTimeout(sendData, 2000);
        }
      } catch (err) {
        sendTimeout = setTimeout(sendData, 2000);
      }
    };

    // Initial send and interval setup
    sendData();
    sendInterval = setInterval(sendData, 500);

    // Heartbeat mechanism
    pingInterval = setInterval(() => {
      if (connection.socket.readyState === 1) {
        connection.socket.ping();
      }
    }, 25000);

    // Cleanup function
    const cleanup = () => {
      clearInterval(sendInterval);
      clearInterval(pingInterval);
      if (sendTimeout) clearTimeout(sendTimeout);
      [
        'exit', 'SIGINT', 'SIGTERM',
        'uncaughtException', 'unhandledRejection'
      ].forEach(event => process.off(event, cleanup));
      
      if (connection.socket.readyState === 1) {
        connection.socket.close();
      }
    };

    // Process event handlers
    process.on('exit', cleanup);
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    process.on('uncaughtException', cleanup);
    process.on('unhandledRejection', cleanup);

    // WebSocket close handler
    connection.socket.on('close', () => {
      console.log('Client disconnected');
      cleanup();
    });
  });
});


fastify.register(async function (fastify) {
  fastify.get(
    "/price-data",
    { websocket: true },
    (connection, req) => {
      let tickers = [];
      let sendInterval;
      // Store the last sent data for each ticker
      const lastSentData = {};
      
      // Function to send data for all tickers as a list
const sendData = async () => {
  const dataToSend = [];

  for (const symbol of tickers) {
    const filePath = path?.join(
      __dirname,
      `../app/json/websocket/companies/${symbol?.toUpperCase()}.json`
    );

    try {
      if (fs?.existsSync(filePath)) {
        const fileData = fs?.readFileSync(filePath, "utf8");

        if (!fileData) {
          console.error(`File is empty for ticker: ${symbol}`);
          continue;
        }

        let jsonData;
        try {
          jsonData = JSON?.parse(fileData);
        } catch (parseError) {
          //console.error(`Invalid JSON format for ticker: ${symbol}`, parseError);
          //console.error(`File content: ${fileData}`);
          continue;
        }

        // Only send data if conditions are met and data has changed
        if (
          jsonData?.lp != null &&
          jsonData?.ap != null &&
          jsonData?.bp != null &&
          jsonData?.t != null &&
          ["Q", "T"]?.includes(jsonData?.type) &&
          connection.socket.readyState === WebSocket.OPEN
        ) {
          const avgPrice = (
            parseFloat(jsonData?.ap) +
            parseFloat(jsonData?.bp) +
            parseFloat(jsonData?.lp)
          ) / 3;

          const finalPrice = Math.abs(avgPrice - jsonData?.bp) / jsonData?.bp > 0.01
            ? jsonData.bp 
            : avgPrice;

          // Check if finalPrice is equal to avgPrice before sending data
          if (finalPrice - avgPrice === 0) {
            const currentDataSignature = `${finalPrice}`;
            const lastSentSignature = lastSentData[symbol];

            if (currentDataSignature !== lastSentSignature) {
              dataToSend?.push({
                symbol,
                ap: jsonData?.ap,
                bp: jsonData?.bp,
                lp: jsonData?.lp,
                avgPrice: finalPrice,
                type: jsonData?.type,
                time: formatTimestampNewYork(jsonData?.t),
              });

              lastSentData[symbol] = currentDataSignature;
            }
          }
        }
      } else {
        //console.error("File not found for ticker:", symbol);
      }
    } catch (err) {
      //console.error("Error processing data for ticker:", symbol, err);
    }
  }

  // Send all collected data as a single message
  if (dataToSend.length > 0 && connection.socket.readyState === WebSocket.OPEN) {
    connection.socket.send(JSON.stringify(dataToSend));
    // console.log(dataToSend);
  }
};


      
      // Start receiving messages from the client
      connection.socket.on("message", (message) => {
        try {
          // Parse message as JSON to get tickers array
          tickers = JSON.parse(message.toString("utf-8"));
          //console.log("Received tickers from client:", tickers);
          
          // Reset last sent data for new tickers
          tickers?.forEach((ticker) => {
            lastSentData[ticker] = null;
          });
          
          // Start periodic data sending if not already started
          if (!sendInterval) {
            sendInterval = setInterval(sendData, 1000);
          }
        } catch (err) {
          console.error("Failed to parse tickers from client message:", err);
        }
      });
      
      // Handle client disconnect
      connection.socket.on("close", () => {
        console.log("Client disconnected");
        clearInterval(sendInterval);
        removeProcessListeners();
      });
      
      // Handle server crash cleanup
      const closeHandler = () => {
        console.log("Server is closing. Cleaning up resources...");
        clearInterval(sendInterval);
        connection.socket.close();
        removeProcessListeners();
      };
      
      // Add close handler to process events
      process.on("exit", closeHandler);
      process.on("SIGINT", closeHandler);
      process.on("SIGTERM", closeHandler);
      process.on("uncaughtException", closeHandler);
      process.on("unhandledRejection", closeHandler);
      
      // Function to remove process listeners to avoid memory leaks
      const removeProcessListeners = () => {
        process.off("exit", closeHandler);
        process.off("SIGINT", closeHandler);
        process.off("SIGTERM", closeHandler);
        process.off("uncaughtException", closeHandler);
        process.off("unhandledRejection", closeHandler);
      };
    }
  );
});



// Function to start the server
function startServer() {
  if (!serverRunning) {
    fastify.listen(2000, (err) => {
      if (err) {
        console.error("Error starting server:", err);
        process.exit(1); // Exit the process if server start fails
      }
      serverRunning = true;
      console.log("Server started successfully on port 2000!");
    });
  } else {
    console.log("Server is already running.");
  }
}

// Function to stop the server
function stopServer() {
  if (serverRunning) {
    return new Promise((resolve, reject) => {
      fastify.close((err) => {
        if (err) {
          console.error("Error closing server:", err);
          reject(err);
        } else {
          serverRunning = false;
          console.log("Server closed successfully!");
          resolve();
        }
      });
    });
  } else {
    console.log("Server is not running.");
    return Promise.resolve();
  }
}

// Function to gracefully close and restart the server
function restartServer() {
  if (serverRunning) {
    stopServer()
      .then(() => {
        console.log("Restarting server...");
        startServer();
      })
      .catch((error) => {
        console.error("Failed to restart server:", error);
        process.exit(1); // Exit the process if server restart fails
      });
  } else {
    console.log("Server is not running. Starting server...");
    startServer();
  }
}

// Add a global error handler for uncaught exceptions
process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:", err);
  restartServer();
});

// Add a global error handler for unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
  restartServer();
});

// Start the server
startServer();
