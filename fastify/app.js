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
    "/realtime-data",
    { websocket: true },
    (connection, req) => {

      let jsonData;
      let sendInterval;
      let symbol;
      let isSend = false;

     
      // Function to send data to the client
     const sendData = async () => {
  if (!symbol) return; // Check if symbol is defined
  const filePath = path.join(__dirname, `../app/json/websocket/companies/${symbol}.json`);

  try {
    if (fs.existsSync(filePath)) {
      const fileData = fs.readFileSync(filePath, "utf8");
      jsonData = JSON.parse(fileData);

      // Logic to send data if certain conditions are met
      if (
        jsonData?.lp != null &&
        jsonData?.t != null &&
        ["Q", "T"].includes(jsonData?.type) &&
        connection.socket.readyState === WebSocket.OPEN &&
        !isSend
      ) {
        // Calculate the average price
        const avgPrice =
          (parseFloat(jsonData.ap) +
            parseFloat(jsonData.bp) +
            parseFloat(jsonData.lp)) /
          3;

        connection.socket.send(
          JSON.stringify({
            bp: jsonData?.bp,
            ap: jsonData?.ap,
            lp: jsonData?.lp?.toFixed(2),
            avgPrice: avgPrice?.toFixed(2), // Add the computed average price
            type: jsonData?.type,
            time: formatTimestampNewYork(jsonData?.t),
          })
        );

        isSend = true;
        setTimeout(() => {
          isSend = false;
        }, 500); // Reset isSend after 500ms
      }
    } else {
      console.error("File not found:", filePath);
      clearInterval(sendInterval);
      connection.socket.close();
      console.error("Connection closed");
    }
  } catch (err) {
    console.error("Error sending data to client:", err);
    clearInterval(sendInterval);
    connection.socket.close();
  }
};


      // Start receiving messages from the client
      connection.socket.on("message", (message) => {
        symbol = message.toString("utf-8")?.toUpperCase();
        console.log("Received message from client:", symbol);

        // Send data immediately upon receiving a symbol
        sendData();

        // Start periodic data sending if not already started
        if (!sendInterval) {
          sendInterval = setInterval(sendData, 1000);
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
  fastify.get(
    "/options-flow-reader",
    { websocket: true },
    (connection, req) => {
      let jsonData;
      let sendInterval;

      // Function to send data to the client
      const sendData = async () => {
        const filePath = path.join(
          __dirname,
          "../app/json/options-flow/feed/data.json"
        );
        try {
          if (fs.existsSync(filePath)) {
            const fileData = fs.readFileSync(filePath, "utf8");
            jsonData = JSON.parse(fileData);
            connection.socket.send(JSON.stringify(jsonData));
          } else {
            console.error("File not found:", filePath);
            clearInterval(sendInterval);
            connection?.socket?.close();
            console.error("Connection closed");
            throw new Error("This is an intentional uncaught exception!");
          }
        } catch (err) {
          console.error("Error sending data to client:", err);
        }
      };

      // Send data to the client initially
      sendData();

      // Start sending data periodically
      sendInterval = setInterval(sendData, 5000);

      // Handle client disconnect
      connection.socket.on("close", () => {
        console.log("Client disconnected");
        clearInterval(sendInterval);
      });

      // Handle server crash cleanup
      const closeHandler = () => {
        console.log("Server is closing. Cleaning up resources...");
        clearInterval(sendInterval);
        connection.socket.close();
      };

      // Add close handler to process event
      process.on("exit", closeHandler);
      process.on("SIGINT", closeHandler);
      process.on("SIGTERM", closeHandler);
      process.on("uncaughtException", closeHandler);
      process.on("unhandledRejection", closeHandler);
    }
  );
});



fastify.register(async function (fastify) {
  fastify.get(
    "/multiple-realtime-data",
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
      `../app/json/websocket/companies/${symbol}.json`
    );

    try {
      if (fs?.existsSync(filePath)) {
        const fileData = fs?.readFileSync(filePath, "utf8");
        const jsonData = JSON?.parse(fileData);

        // Only send data if conditions are met and data has changed
        if (
          jsonData?.lp != null &&
          jsonData?.ap != null &&
          jsonData?.bp != null &&
          jsonData?.t != null &&
          ["Q", "T"].includes(jsonData?.type) &&
          connection.socket.readyState === WebSocket.OPEN
        ) {
          // Calculate the average price
          const avgPrice =
            ((parseFloat(jsonData.ap) +
              parseFloat(jsonData.bp) +
              parseFloat(jsonData.lp)) /
            3);

          // Check if the current data is different from the last sent data
          const currentDataSignature = `${jsonData?.lp}`;
          const lastSentSignature = lastSentData[symbol];

          if (currentDataSignature !== lastSentSignature) {
            // Collect data to send
            dataToSend?.push({
              symbol, // Include the ticker symbol in the sent data
              ap: jsonData?.ap,
              bp: jsonData?.bp,
              lp: jsonData?.lp,
              avgPrice: avgPrice, // Add the computed average price
            });

            // Update the last sent data for this ticker
            lastSentData[symbol] = currentDataSignature;
          }
        }
      } else {
        //console.error("File not found for ticker:", symbol);
      }
    } catch (err) {
      console.error("Error processing data for ticker:", symbol, err);
    }
  }

  // Send all collected data as a single message
  if (dataToSend.length > 0 && connection.socket.readyState === WebSocket.OPEN) {
    connection.socket.send(JSON.stringify(dataToSend));
    //console.log(dataToSend);
  }
};

      
      // Start receiving messages from the client
      connection.socket.on("message", (message) => {
        try {
          // Parse message as JSON to get tickers array
          tickers = JSON.parse(message.toString("utf-8"));
          console.log("Received tickers from client:", tickers);
          
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
