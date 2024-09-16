const fastify = require("fastify")({ logger: true });
const cors = require("@fastify/cors");
const WebSocket = require("ws");
const path = require("path");
const fs = require("fs");
const PocketBase = require("pocketbase/cjs");
const { serialize } = require("object-to-formdata");
const got = require("got");
const cheerio = require("cheerio");
const sharp = require("sharp");
const axios = require("axios");

let serverRunning = false;
const activeConnections = new Set();

// Load API KEYS
require("dotenv").config({ path: "../app/.env" });
const fmpAPIKey = process.env.FMP_API_KEY;
const twitchAPIKey = process.env.TWITCH_API_KEY;
const twitchSecretKey = process.env.TWITCH_SECRET_KEY;

const pb = new PocketBase("http://127.0.0.1:8090");
pb.autoCancellation(false);

// Register the CORS plugin
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

// Register routes
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

fastify.register(require("@fastify/websocket"));

fastify.register(async function (fastify) {
  fastify.get("/realtime-data", { websocket: true }, handleWebSocket);
  fastify.get(
    "/realtime-crypto-data",
    { websocket: true },
    handleCryptoWebSocket
  );
  fastify.get(
    "/options-flow-reader",
    { websocket: true },
    handleOptionsFlowWebSocket
  );
});

function handleWebSocket(connection, req) {
  let symbol = "";
  let ws;

  const cleanup = () => {
    activeConnections.delete(connection);
    if (ws) ws.close();
  };

  activeConnections.add(connection);

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
      fastify.log.info("Connected to FMP WebSocket");
      ws.send(JSON.stringify(login));
      setTimeout(() => updateSubscription(), 2000);
    });

    ws.on("error", function (error) {
      fastify.log.error("FMP WebSocket error:", error);
    });

    ws.on("close", function () {
      fastify.log.info("FMP WebSocket closed. Attempting to reconnect...");
      setTimeout(connectToFMP, 5000);
    });

    ws.on("message", function (data) {
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
        fastify.log.error("Error parsing JSON:", error);
      }
    });
  }

  connectToFMP();

  connection.socket.on("message", (message) => {
    symbol = message.toString("utf-8");
    fastify.log.info("Received message from client:", symbol);
    updateSubscription();
  });

  connection.socket.on("close", () => {
    fastify.log.info("Client disconnected");
    cleanup();
  });

  connection.socket.on("error", (error) => {
    fastify.log.error("WebSocket error:", error);
    cleanup();
  });
}

function handleCryptoWebSocket(connection, req) {
  let symbol = "";
  let ws;

  const cleanup = () => {
    activeConnections.delete(connection);
    if (ws) ws.close();
  };

  activeConnections.add(connection);

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
      fastify.log.info("Connected to Crypto FMP WebSocket");
      ws.send(JSON.stringify(login));
      setTimeout(() => updateSubscription(), 2000);
    });

    ws.on("error", function (error) {
      fastify.log.error("Crypto FMP WebSocket error:", error);
    });

    ws.on("close", function () {
      fastify.log.info(
        "Crypto FMP WebSocket closed. Attempting to reconnect..."
      );
      setTimeout(connectToCryptoFMP, 5000);
    });

    ws.on("message", function (data) {
      const stringData = data.toString("utf-8");
      if (connection.socket.readyState === WebSocket.OPEN) {
        connection.socket.send(stringData);
      }
    });
  }

  connectToCryptoFMP();

  connection.socket.on("message", (message) => {
    symbol = message.toString("utf-8");
    fastify.log.info("Received message from client:", symbol);
    updateSubscription();
  });

  connection.socket.on("close", () => {
    fastify.log.info("Client disconnected");
    cleanup();
  });

  connection.socket.on("error", (error) => {
    fastify.log.error("WebSocket error:", error);
    cleanup();
  });
}

function handleOptionsFlowWebSocket(connection, req) {
  let jsonData;
  let sendInterval;
  let clientIds = [];

  const cleanup = () => {
    activeConnections.delete(connection);
    clearInterval(sendInterval);
  };

  activeConnections.add(connection);

  const sendData = async () => {
    const filePath = path.join(
      __dirname,
      "../app/json/options-flow/feed/data.json"
    );
    try {
      if (fs.existsSync(filePath)) {
        const fileData = fs.readFileSync(filePath, "utf8");
        jsonData = JSON.parse(fileData);

        if (clientIds.length === 0) {
          fastify.log.info("Client IDs list is empty, doing nothing.");
          return;
        }

        const filteredData = jsonData.filter(
          (item) => !clientIds.includes(item.id)
        );

        if (connection.socket.readyState === WebSocket.OPEN) {
          connection.socket.send(JSON.stringify(filteredData));
        }
      } else {
        fastify.log.error("File not found:", filePath);
        clearInterval(sendInterval);
      }
    } catch (err) {
      fastify.log.error("Error sending data to client:", err);
    }
  };

  sendInterval = setInterval(sendData, 5000);

  connection.socket.on("message", (message) => {
    try {
      const parsedMessage = JSON.parse(message);
      if (parsedMessage?.ids) {
        clientIds = parsedMessage.ids;
      }
    } catch (error) {
      fastify.log.error("Error parsing incoming message:", error);
    }
  });

  connection.socket.on("close", () => {
    fastify.log.info("Client disconnected");
    cleanup();
  });

  connection.socket.on("error", (error) => {
    fastify.log.error("WebSocket error:", error);
    cleanup();
  });

  sendData();
}

// Graceful shutdown
async function gracefulShutdown() {
  fastify.log.info("Shutting down gracefully...");
  for (const connection of activeConnections) {
    connection.socket.close();
  }
  await fastify.close();
  process.exit(0);
}

process.on("SIGINT", gracefulShutdown);
process.on("SIGTERM", gracefulShutdown);

// Error handling
fastify.setErrorHandler((error, request, reply) => {
  fastify.log.error(error);
  reply.status(500).send({ error: "Internal Server Error" });
});

// Start the server
async function startServer() {
  try {
    await fastify.listen({ port: 2000, host: "0.0.0.0" });
    serverRunning = true;
    fastify.log.info("Server started successfully on port 2000!");
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

// Function to stop the server
async function stopServer() {
  if (serverRunning) {
    try {
      await fastify.close();
      serverRunning = false;
      fastify.log.info("Server closed successfully!");
    } catch (err) {
      fastify.log.error("Error closing server:", err);
      throw err;
    }
  } else {
    fastify.log.info("Server is not running.");
  }
}

// Function to gracefully close and restart the server
async function restartServer() {
  if (serverRunning) {
    try {
      await stopServer();
      fastify.log.info("Restarting server...");
      await startServer();
    } catch (error) {
      fastify.log.error("Failed to restart server:", error);
      process.exit(1);
    }
  } else {
    fastify.log.info("Server is not running. Starting server...");
    await startServer();
  }
}

// Global error handlers
process.on("uncaughtException", (err) => {
  fastify.log.error("Uncaught Exception:", err);
  restartServer();
});

process.on("unhandledRejection", (reason, promise) => {
  fastify.log.error("Unhandled Rejection at:", promise, "reason:", reason);
  restartServer();
});

// Start the server
startServer();
