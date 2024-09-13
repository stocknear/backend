let serverRunning = false;

const fastify = require("fastify")({});
const cors = require("@fastify/cors");

//Load API KEYS
require("dotenv").config({ path: "../app/.env" });
const fmpAPIKey = process.env.FMP_API_KEY;
//const mixpanelAPIKey =  process.env.MIXPANEL_API_KEY;
const twitchAPIKey = process.env.TWITCH_API_KEY;
const twitchSecretKey = process.env.TWITCH_SECRET_KEY;

//const Mixpanel = require('mixpanel');
//const UAParser = require('ua-parser-js');

const got = require("got"); //Only version npm i got@11.8.3 works with ESM
const cheerio = require("cheerio");
const sharp = require("sharp");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
//const pino = require('pino');

//const mixpanel = Mixpanel.init(mixpanelAPIKey, { debug: false });

const PocketBase = require("pocketbase/cjs");
const pb = new PocketBase("http://127.0.0.1:8090");

// globally disable auto cancellation
//See https://github.com/pocketbase/js-sdk#auto-cancellation
//Bug happens that get-post gives an error of auto-cancellation. Hence set it to false;
pb.autoCancellation(false);

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

//fastify.register(require('./mixpanel/server'), { mixpanel, UAParser });
fastify.register(require("./get-user-stats/server"), { pb });
fastify.register(require("./get-community-stats/server"), { pb });
fastify.register(require("./get-moderators/server"), { pb });
fastify.register(require("./get-user-data/server"), { pb });
fastify.register(require("./get-all-comments/server"), { pb });
fastify.register(require("./get-post/server"), { pb });
fastify.register(require("./get-one-post/server"), { pb });
fastify.register(require("./update-watchlist/server"), { pb, serialize });
fastify.register(require("./get-portfolio-data/server"), { pb });
fastify.register(require("./create-portfolio/server"), { pb, serialize });
fastify.register(require("./buy-stock/server"), { pb });
fastify.register(require("./sell-stock/server"), { pb });
fastify.register(require("./create-post-link/server"), { got, cheerio, sharp });
fastify.register(require("./create-post-image/server"), { sharp });
fastify.register(require("./delete-comment/server"), { pb });
fastify.register(require("./delete-post/server"), { pb });
fastify.register(require("./leaderboard/server"), { pb });
fastify.register(require("./feedback/server"), { pb });
fastify.register(require("./create-watchlist/server"), { pb });
fastify.register(require("./delete-watchlist/server"), { pb });
fastify.register(require("./edit-name-watchlist/server"), { pb });
fastify.register(require("./all-watchlists/server"), { pb });
fastify.register(require("./get-notifications/server"), { pb });
fastify.register(require("./update-notifications/server"), { pb });
fastify.register(require("./create-strategy/server"), { pb });
fastify.register(require("./delete-strategy/server"), { pb });
fastify.register(require("./all-strategies/server"), { pb });
fastify.register(require("./save-strategy/server"), { pb });
fastify.register(require("./get-strategy/server"), { pb });
fastify.register(require("./get-twitch-status/server"), {
  axios,
  twitchAPIKey,
  twitchSecretKey,
});
fastify.register(require("./get-portfolio/server"), { pb });
fastify.register(require("./create-price-alert/server"), { pb });
fastify.register(require("./get-price-alert/server"), { pb, fs, path });
fastify.register(require("./delete-price-alert/server"), { pb });
fastify.register(require("./upvote/server"), { pb });
fastify.register(require("./downvote/server"), { pb });
fastify.register(require("./upvote-comment/server"), { pb });
fastify.register(require("./downvote-comment/server"), { pb });

//fastify.register(require('./create-comment/server'), { pb });

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

fastify.register(async function (fastify) {
  fastify.get("/realtime-data", { websocket: true }, (connection, req) => {
    // Send a welcome message to the client

    //connection.socket.send('hi from server');

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
    const ws = new WebSocket("wss://websockets.financialmodelingprep.com");

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

    ws.on("message", function (data, flags) {
      const stringData = data.toString("utf-8");

      try {
        const jsonData = JSON.parse(stringData);
        const bpData = jsonData.bp;

        // Check if bpData is a number and not equal to zero
        if (typeof bpData === "number" && bpData !== 0) {
          if (connection.socket.readyState === WebSocket.OPEN && !isSend) {
            connection.socket.send(JSON.stringify({ bp: bpData }));
            isSend = true;
            setTimeout(() => {
              isSend = false;
            }, 800);
          }
        }
      } catch (error) {
        console.error("Error parsing JSON:", error);
      }
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
  });
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
          }, 800);

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
      let clientIds = [];

      // Function to send filtered data to the client
      const sendData = async () => {
        const filePath = path.join(
          __dirname,
          "../app/json/options-flow/feed/data.json"
        );
        try {
          if (fs.existsSync(filePath)) {
            const fileData = fs.readFileSync(filePath, "utf8");
            jsonData = JSON.parse(fileData);

            // Do nothing if clientIds is empty
            if (clientIds.length === 0) {
              console.log("Client IDs list is empty, doing nothing.");
              return; // Exit function if clientIds is empty
            }

            // Filter out elements whose ids are not in clientIds
            const filteredData = jsonData.filter(
              (item) => !clientIds.includes(item.id)
            );

            // Send the filtered data back to the client
            connection.socket.send(JSON.stringify(filteredData));
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

      // Handle incoming messages from the client to update the ids
      connection.socket.on("message", (message) => {
        try {
          const parsedMessage = JSON.parse(message);
          if (parsedMessage?.ids) {
            //console.log("Received ids from client:", parsedMessage.ids);
            clientIds = parsedMessage.ids; // Update the ids list from the client
          } else {
            //console.log("No ids received in the message");
          }
        } catch (error) {
          console.error("Error parsing incoming message:", error);
        }
      });

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
