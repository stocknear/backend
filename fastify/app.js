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
    let symbol = "";
    let isConnectionActive = true;
    let ws;

    const login = {
      event: "login",
      data: {
        apiKey: fmpAPIKey,
      },
    };

    const subscribe = {
      event: "subscribe",
      data: {
        ticker: "",
      },
    };

    function updateSubscription() {
      subscribe.data.ticker = symbol;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(subscribe));
      }
    }

    function connectToFMP() {
      ws = new WebSocket("wss://websockets.financialmodelingprep.com");

      ws.on("open", function open() {
        console.log("Connected to FMP WebSocket");
        ws.send(JSON.stringify(login));
        setTimeout(() => updateSubscription(), 2000);
      });

      ws.on("error", function (error) {
        console.error("FMP WebSocket error:", error);
      });

      ws.on("close", function () {
        console.log("FMP WebSocket closed. Attempting to reconnect...");
        setTimeout(connectToFMP, 5000);
      });

      ws.on("message", function (data) {
        if (!isConnectionActive) return;

        const stringData = data.toString("utf-8");
        try {
          const jsonData = JSON.parse(stringData);
          const bpData = jsonData.bp;

          if (typeof bpData === "number" && bpData !== 0) {
            if (connection.socket.readyState === WebSocket.OPEN) {
              connection.socket.send(JSON.stringify({ bp: bpData }));
            }
          }
        } catch (error) {
          console.error("Error parsing JSON:", error);
        }
      });
    }

    connectToFMP();

    connection.socket.on("message", (message) => {
      symbol = message.toString("utf-8");
      console.log("Received message from client:", symbol);
      updateSubscription();
    });

    connection.socket.on("close", () => {
      console.log("Client disconnected");
      isConnectionActive = false;
      if (ws) ws.close();
    });

    connection.socket.on("error", (error) => {
      console.error("WebSocket error:", error);
      isConnectionActive = false;
      if (ws) ws.close();
    });
  });
});

fastify.register(async function (fastify) {
  fastify.get(
    "/realtime-crypto-data",
    { websocket: true },
    (connection, req) => {
      let symbol = "";
      let isConnectionActive = true;
      let ws;

      const login = {
        event: "login",
        data: {
          apiKey: fmpAPIKey,
        },
      };

      const subscribe = {
        event: "subscribe",
        data: {
          ticker: "",
        },
      };

      function updateSubscription() {
        subscribe.data.ticker = symbol;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(subscribe));
        }
      }

      function connectToCryptoFMP() {
        ws = new WebSocket("wss://crypto.financialmodelingprep.com");

        ws.on("open", function open() {
          console.log("Connected to Crypto FMP WebSocket");
          ws.send(JSON.stringify(login));
          setTimeout(() => updateSubscription(), 2000);
        });

        ws.on("error", function (error) {
          console.error("Crypto FMP WebSocket error:", error);
        });

        ws.on("close", function () {
          console.log(
            "Crypto FMP WebSocket closed. Attempting to reconnect..."
          );
          setTimeout(connectToCryptoFMP, 5000);
        });

        ws.on("message", function (data) {
          if (!isConnectionActive) return;

          const stringData = data.toString("utf-8");
          if (connection.socket.readyState === WebSocket.OPEN) {
            connection.socket.send(stringData);
          }
        });
      }

      connectToCryptoFMP();

      connection.socket.on("message", (message) => {
        symbol = message.toString("utf-8");
        console.log("Received message from client:", symbol);
        updateSubscription();
      });

      connection.socket.on("close", () => {
        console.log("Client disconnected");
        isConnectionActive = false;
        if (ws) ws.close();
      });

      connection.socket.on("error", (error) => {
        console.error("WebSocket error:", error);
        isConnectionActive = false;
        if (ws) ws.close();
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
      let isConnectionActive = true;

      // Function to send filtered data to the client
      const sendData = async () => {
        if (!isConnectionActive) return;

        const filePath = path.join(
          __dirname,
          "../app/json/options-flow/feed/data.json"
        );
        try {
          if (fs.existsSync(filePath)) {
            const fileData = fs.readFileSync(filePath, "utf8");
            jsonData = JSON.parse(fileData);

            if (clientIds.length === 0) {
              console.log("Client IDs list is empty, doing nothing.");
              return;
            }

            const filteredData = jsonData.filter(
              (item) => !clientIds.includes(item.id)
            );

            if (connection.socket.readyState === WebSocket.OPEN) {
              connection.socket.send(JSON.stringify(filteredData));
            }
          } else {
            console.error("File not found:", filePath);
            clearInterval(sendInterval);
          }
        } catch (err) {
          console.error("Error sending data to client:", err);
          // Don't close the connection here, just log the error
        }
      };

      // Start sending data periodically
      sendInterval = setInterval(sendData, 5000);

      // Handle incoming messages from the client to update the ids
      connection.socket.on("message", (message) => {
        try {
          const parsedMessage = JSON.parse(message);
          if (parsedMessage?.ids) {
            clientIds = parsedMessage.ids;
          }
        } catch (error) {
          console.error("Error parsing incoming message:", error);
        }
      });

      // Handle client disconnect
      connection.socket.on("close", () => {
        console.log("Client disconnected");
        isConnectionActive = false;
        clearInterval(sendInterval);
      });

      // Handle WebSocket errors
      connection.socket.on("error", (error) => {
        console.error("WebSocket error:", error);
        isConnectionActive = false;
        clearInterval(sendInterval);
        // Don't close the connection here, let the client handle reconnection
      });

      // Send initial data
      sendData();
    }
  );
});

fastify.setErrorHandler((error, request, reply) => {
  console.error("Server error:", error);
  reply.status(500).send({ error: "Internal Server Error" });
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("Received SIGINT. Shutting down gracefully...");
  await fastify.close();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  console.log("Received SIGTERM. Shutting down gracefully...");
  await fastify.close();
  process.exit(0);
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
