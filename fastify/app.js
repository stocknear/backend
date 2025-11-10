let serverRunning = false;

const fastify = require("fastify")({
  logger: process.env.NODE_ENV === "development",
});
const cors = require("@fastify/cors");
const websocketPlugin = require("@fastify/websocket");

const fsp = require("fs").promises;
const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "../app/.env") });

const WebSocket = require("ws");

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

function corsMiddleware(request, reply, done) {
  const origin = request?.headers?.origin;
  if (!origin || allowedOrigins.includes(origin)) {
    reply.header("Access-Control-Allow-Origin", origin || "*");
    reply.header("Access-Control-Allow-Methods", "GET,POST");
    reply.header("Access-Control-Allow-Headers", "Content-Type");
    done();
  } else {
    reply.code(403).send({ error: "Forbidden" });
  }
}

fastify.register(cors);
fastify.addHook("onRequest", corsMiddleware);

fastify.register(websocketPlugin, {
  options: {
    maxPayload: 1024 * 1024,
    perMessageDeflate: true,
  },
});

function isSocketOpen(socket) {
  return socket && socket.readyState === WebSocket.OPEN;
}

function sendRaw(socket, message) {
  if (!isSocketOpen(socket)) {
    return;
  }
  try {
    socket.send(message);
  } catch (err) {
    console.error("Failed to send WebSocket message:", err);
  }
}

function sendJson(socket, payload) {
  if (!isSocketOpen(socket)) {
    return;
  }
  try {
    socket.send(JSON.stringify(payload));
  } catch (err) {
    console.error("Failed to send WebSocket message:", err);
  }
}

function formatTimestampNewYork(timestamp) {
  const numericTimestamp = Number(timestamp);
  if (!Number.isFinite(numericTimestamp)) {
    return "";
  }
  const d = new Date(numericTimestamp / 1e6);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
    ?.format(d)
    ?.replace(/(\d+)\/(\d+)\/(\d+),/, "$3-$1-$2")
    ?.replace(",", "");
}

function roundToTwo(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Number(number.toFixed(2)) : null;
}

function toFiniteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

const OPTIONS_FLOW_FIELDS_TO_REMOVE = [
  "exchange",
  "tradeCount",
  "description",
  "aggressor_ind",
  "ask",
  "bid",
  "midpoint",
  "trade_count",
];
const MAX_OPTIONS_FLOW_HISTORY = 2000;
const OPTIONS_FLOW_FILE_PATH = path.join(
  __dirname,
  "../app/json/options-flow/feed/data.json",
);
const OPTIONS_FLOW_POLL_INTERVAL_MS = 3000;

const optionsFlowSubscribers = new Set();
let optionsFlowTimer = null;
let optionsFlowCache = [];
let optionsFlowRawSignature = null;

function ensureOptionsFlowPolling() {
  if (!optionsFlowTimer) {
    optionsFlowTimer = setInterval(() => {
      pollOptionsFlow().catch((err) => {
        console.error("Error polling options flow data:", err);
      });
    }, OPTIONS_FLOW_POLL_INTERVAL_MS);
  }
}

function stopOptionsFlowPollingIfIdle() {
  if (optionsFlowTimer && optionsFlowSubscribers.size === 0) {
    clearInterval(optionsFlowTimer);
    optionsFlowTimer = null;
    optionsFlowCache = [];
    optionsFlowRawSignature = null;
  }
}

async function pollOptionsFlow() {
  if (optionsFlowSubscribers.size === 0) {
    return;
  }

  try {
    const fileData = await fsp.readFile(OPTIONS_FLOW_FILE_PATH, "utf8");
    if (!fileData) {
      return;
    }
    if (fileData === optionsFlowRawSignature) {
      return;
    }

    let parsed;
    try {
      parsed = JSON.parse(fileData);
    } catch (err) {
      console.error("Invalid JSON format for options flow data:", err);
      return;
    }

    let cleaned = Array.isArray(parsed)
      ? parsed
          .map((item) => cleanOptionsFlowItem(item))
          .filter((item) => item !== null)
      : [];

    if (cleaned.length > 10000) {
      cleaned = cleaned.filter((item) => meetsOptionsFlowThreshold(item));
    }

    optionsFlowCache = cleaned;
    optionsFlowRawSignature = fileData;

    for (const subscriber of optionsFlowSubscribers) {
      deliverOptionsFlow(subscriber, cleaned);
    }
  } catch (err) {
    if (err.code !== "ENOENT") {
      console.error("Error reading options flow data file:", err);
    }
  }
}

function cleanOptionsFlowItem(item) {
  if (!item || typeof item !== "object") {
    return null;
  }
  const cleanedItem = {};
  for (const key of Object.keys(item)) {
    if (!OPTIONS_FLOW_FIELDS_TO_REMOVE.includes(key)) {
      cleanedItem[key] = item[key];
    }
  }
  if (cleanedItem.id == null) {
    return null;
  }
  return cleanedItem;
}

function meetsOptionsFlowThreshold(item) {
  if (!item) {
    return false;
  }
  const volume = toFiniteNumber(item.volume);
  const hasCostBasis =
    item.cost_basis !== null &&
    item.cost_basis !== undefined &&
    item.cost_basis !== "" &&
    item.cost_basis !== 0;
  const premiumSource = hasCostBasis ? item.cost_basis : item.premium;
  const premium = toFiniteNumber(premiumSource);
  return volume >= 100 && premium >= 50000;
}

function deliverOptionsFlow(subscriber, data) {
  if (!isSocketOpen(subscriber.socket) || !Array.isArray(data) || data.length === 0) {
    return;
  }

  const payload = [];
  for (const item of data) {
    const id = item.id;
    if (id == null) {
      continue;
    }
    if (subscriber.orderList.has(id)) {
      continue;
    }
    if (subscriber.sentIds.has(id)) {
      continue;
    }
    payload.push(item);
  }

  if (payload.length === 0) {
    return;
  }

  sendJson(subscriber.socket, payload);
  trackOptionsFlowSent(subscriber, payload.map((entry) => entry.id));
}

function deliverOptionsFlowSnapshot(subscriber) {
  if (optionsFlowCache.length === 0) {
    return;
  }
  deliverOptionsFlow(subscriber, optionsFlowCache);
}

function trackOptionsFlowSent(subscriber, ids) {
  for (const id of ids) {
    subscriber.sentIds.add(id);
    subscriber.sentHistory.push(id);
  }
  while (subscriber.sentHistory.length > MAX_OPTIONS_FLOW_HISTORY) {
    const oldest = subscriber.sentHistory.shift();
    if (oldest !== undefined) {
      subscriber.sentIds.delete(oldest);
    }
  }
}

function updateSubscriberOrderList(subscriber, list) {
  const nextIds = Array.isArray(list) ? list : [];
  subscriber.orderList = new Set(nextIds);
  if (subscriber.sentHistory.length > 0) {
    subscriber.sentHistory = subscriber.sentHistory.filter(
      (id) => !subscriber.orderList.has(id),
    );
  }
  for (const id of subscriber.orderList) {
    subscriber.sentIds.delete(id);
  }
}

function cleanupOptionsFlowSubscriber(subscriber) {
  optionsFlowSubscribers.delete(subscriber);
  stopOptionsFlowPollingIfIdle();
}

function handleOptionsFlow(connection) {
  const subscriber = {
    socket: connection.socket,
    orderList: new Set(),
    sentIds: new Set(),
    sentHistory: [],
  };

  optionsFlowSubscribers.add(subscriber);
  ensureOptionsFlowPolling();

  const handleMessage = (message) => {
    try {
      const payload = JSON.parse(message.toString("utf-8"));
      if (
        payload &&
        (payload.type === "init" || payload.type === "update")
      ) {
        updateSubscriberOrderList(subscriber, payload.orderList);
        deliverOptionsFlowSnapshot(subscriber);
      }
    } catch (err) {
      console.error("Failed to parse options flow message from client:", err);
    }
  };

  let cleaned = false;
  const cleanup = () => {
    if (cleaned) {
      return;
    }
    cleaned = true;
    cleanupOptionsFlowSubscriber(subscriber);
  };

  connection.socket.on("message", handleMessage);
  connection.socket.on("close", cleanup);
  connection.socket.on("error", (err) => {
    console.error("Options flow WebSocket error:", err);
    cleanup();
  });
}

const PRE_POST_POLL_INTERVAL_MS = 10000;
const PRE_POST_DATA_ROOT = path.join(__dirname, "../app/json/pre-post-quote");
const PRE_POST_MISSING_SIGNATURE = "__missing__";

const prePostRooms = new Map();
const prePostConnections = new Set();
let prePostTimer = null;

function ensurePrePostTimer() {
  if (!prePostTimer && prePostRooms.size > 0) {
    prePostTimer = setInterval(() => {
      runPrePostPipeline().catch((err) => {
        console.error("Error running pre/post pipeline:", err);
      });
    }, PRE_POST_POLL_INTERVAL_MS);
  }
}

function stopPrePostTimerIfIdle() {
  if (prePostTimer && prePostRooms.size === 0) {
    clearInterval(prePostTimer);
    prePostTimer = null;
  }
}

function ensurePrePostRoom(ticker) {
  const upper = ticker.toUpperCase();
  let room = prePostRooms.get(upper);
  if (!room) {
    room = {
      ticker: upper,
      clients: new Set(),
      lastPayload: null,
      lastSignature: null,
      lastMessage: null,
    };
    prePostRooms.set(upper, room);
  }
  return room;
}

function stopPrePostRoomIfEmpty(room) {
  if (room.clients.size === 0) {
    prePostRooms.delete(room.ticker);
  }
}

async function loadPrePostPayloadFromDisk(ticker) {
  const filePath = path.join(PRE_POST_DATA_ROOT, `${ticker}.json`);

  try {
    const fileData = await fsp.readFile(filePath, "utf8");
    if (!fileData) {
      const message = JSON.stringify({});
      return {
        payload: {},
        signature: PRE_POST_MISSING_SIGNATURE,
        message,
      };
    }

    const payload = JSON.parse(fileData);
    const message = JSON.stringify(payload);
    return {
      payload,
      signature: message,
      message,
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      const message = JSON.stringify({});
      return {
        payload: {},
        signature: PRE_POST_MISSING_SIGNATURE,
        message,
      };
    }
    console.error(`Error loading pre-post quote for ${ticker}:`, err);
    return null;
  }
}

async function runPrePostPipeline() {
  if (prePostRooms.size === 0) {
    stopPrePostTimerIfIdle();
    return;
  }

  const symbols = Array.from(prePostRooms.keys());
  const results = await Promise.all(
    symbols.map((symbol) => loadPrePostPayloadFromDisk(symbol)),
  );

  for (let index = 0; index < symbols.length; index += 1) {
    const symbol = symbols[index];
    const room = prePostRooms.get(symbol);
    if (!room) {
      continue;
    }

    const result = results[index];
    if (!result) {
      continue;
    }

    if (room.lastSignature !== result.signature) {
      room.lastPayload = result.payload;
      room.lastSignature = result.signature;
      room.lastMessage = result.message;
      broadcastPrePost(room);
    }
  }
}

function broadcastPrePost(room) {
  if (!room.lastMessage) {
    return;
  }
  for (const client of room.clients) {
    sendRaw(client.socket, room.lastMessage);
  }
}

async function sendPrePostSnapshotForRoom(client, room) {
  if (!room) {
    return;
  }

  if (room.lastMessage) {
    sendRaw(client.socket, room.lastMessage);
    return;
  }

  const loaded = await loadPrePostPayloadFromDisk(room.ticker);
  if (!loaded) {
    return;
  }

  room.lastPayload = loaded.payload;
  room.lastSignature = loaded.signature;
  room.lastMessage = loaded.message;

  sendRaw(client.socket, room.lastMessage);
}

function normalizeTickerSet(values) {
  const normalized = new Set();
  if (!values) {
    return normalized;
  }
  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }
    const trimmed = value.trim().toUpperCase();
    if (trimmed) {
      normalized.add(trimmed);
    }
  }
  return normalized;
}

function extractTickersFromPayload(payload) {
  if (!payload && payload !== "") {
    return null;
  }

  if (Array.isArray(payload)) {
    return normalizeTickerSet(payload);
  }

  if (typeof payload === "string") {
    return normalizeTickerSet([payload]);
  }

  if (payload && typeof payload === "object") {
    if (Array.isArray(payload.tickers)) {
      return normalizeTickerSet(payload.tickers);
    }
    if (Array.isArray(payload.symbols)) {
      return normalizeTickerSet(payload.symbols);
    }
    if (typeof payload.ticker === "string") {
      return normalizeTickerSet([payload.ticker]);
    }
    if (typeof payload.symbol === "string") {
      return normalizeTickerSet([payload.symbol]);
    }
  }

  return null;
}

async function updatePrePostClientSubscriptions(client, tickers) {
  if (!client) {
    return;
  }

  const normalized = normalizeTickerSet(tickers);
  const currentSymbols = Array.from(client.rooms.keys());

  let changed = false;

  for (const symbol of currentSymbols) {
    if (!normalized.has(symbol)) {
      const room = client.rooms.get(symbol);
      if (room) {
        room.clients.delete(client);
        stopPrePostRoomIfEmpty(room);
      }
      client.rooms.delete(symbol);
      changed = true;
    }
  }

  const additions = [];
  for (const symbol of normalized) {
    if (!client.rooms.has(symbol)) {
      const room = ensurePrePostRoom(symbol);
      room.clients.add(client);
      client.rooms.set(symbol, room);
      additions.push(sendPrePostSnapshotForRoom(client, room));
      changed = true;
    }
  }

  if (additions.length > 0) {
    await Promise.all(additions);
  }

  if (prePostRooms.size === 0) {
    stopPrePostTimerIfIdle();
  } else if (changed) {
    ensurePrePostTimer();
  }
}

function removePrePostClient(client) {
  if (!client) {
    return;
  }

  for (const [symbol, room] of client.rooms) {
    if (room) {
      room.clients.delete(client);
      stopPrePostRoomIfEmpty(room);
    }
    client.rooms.delete(symbol);
  }

  prePostConnections.delete(client);

  if (prePostRooms.size === 0) {
    stopPrePostTimerIfIdle();
  }
}

function handlePrePostQuote(connection) {
  const client = {
    socket: connection.socket,
    rooms: new Map(),
  };

  prePostConnections.add(client);

  const cleanup = () => {
    removePrePostClient(client);
  };

  connection.socket.on("message", (message) => {
    let payload;
    try {
      payload = JSON.parse(message.toString("utf-8"));
    } catch (err) {
      payload = message.toString("utf-8");
    }

    const tickers = extractTickersFromPayload(payload);
    if (!tickers) {
      return;
    }

    updatePrePostClientSubscriptions(client, tickers).catch((err) => {
      console.error("Failed to update pre-post subscriptions:", err);
    });
  });

  connection.socket.on("close", cleanup);
  connection.socket.on("error", (err) => {
    console.error("Pre-post quote WebSocket error:", err);
    cleanup();
  });
}

const MARKET_FLOW_POLL_INTERVAL_MS = 10000;
const MARKET_FLOW_DATA_PATH = path.join(
  __dirname,
  "../app/json/market-flow/data.json",
);
const marketFlowClients = new Set();
let marketFlowTimer = null;

function ensureMarketFlowTimer() {
  if (!marketFlowTimer && marketFlowClients.size > 0) {
    marketFlowTimer = setInterval(() => {
      pollMarketFlow().catch((err) => {
        console.error("Error polling market flow data:", err);
      });
    }, MARKET_FLOW_POLL_INTERVAL_MS);
  }
}

function stopMarketFlowTimerIfIdle() {
  if (marketFlowTimer && marketFlowClients.size === 0) {
    clearInterval(marketFlowTimer);
    marketFlowTimer = null;
  }
}

async function loadMarketFlowSnapshot() {
  try {
    const fileData = await fsp.readFile(MARKET_FLOW_DATA_PATH, "utf8");
    if (!fileData) {
      return {
        message: "{}",
      };
    }

    return {
      message: fileData,
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      return {
        message: "{}",
      };
    }
    console.error("Error loading market flow snapshot:", err);
    return null;
  }
}

async function pollMarketFlow() {
  if (marketFlowClients.size === 0) {
    stopMarketFlowTimerIfIdle();
    return;
  }

  const loaded = await loadMarketFlowSnapshot();
  if (!loaded) {
    return;
  }

  broadcastMarketFlow(loaded.message);
}

function broadcastMarketFlow(message) {
  for (const client of marketFlowClients) {
    sendRaw(client.socket, message);
  }
}

async function sendMarketFlowSnapshot(client) {
  if (!client || !isSocketOpen(client.socket)) {
    return;
  }

  const loaded = await loadMarketFlowSnapshot();
  if (!loaded) {
    return;
  }

  sendRaw(client.socket, loaded.message);
}

function removeMarketFlowClient(client) {
  if (!client) {
    return;
  }

  marketFlowClients.delete(client);
  if (marketFlowClients.size === 0) {
    stopMarketFlowTimerIfIdle();
  }
}

function handleMarketFlow(connection) {
  const client = {
    socket: connection.socket,
  };

  marketFlowClients.add(client);
  ensureMarketFlowTimer();

  sendMarketFlowSnapshot(client).catch((err) => {
    console.error("Failed sending market flow snapshot:", err);
  });

  const cleanup = () => {
    removeMarketFlowClient(client);
  };

  connection.socket.on("message", (message) => {
    let payload;
    try {
      payload = JSON.parse(message.toString("utf-8"));
    } catch (err) {
      payload = message.toString("utf-8");
    }

    const isRefreshRequest =
      payload === "refresh" ||
      (payload &&
        typeof payload === "object" &&
        !Array.isArray(payload) &&
        payload.type === "refresh");

    if (isRefreshRequest) {
      pollMarketFlow().catch((err) => {
        console.error("Failed to refresh market flow data:", err);
      });
    }
  });

  connection.socket.on("close", cleanup);
  connection.socket.on("error", (err) => {
    console.error("Market flow WebSocket error:", err);
    cleanup();
  });
}

const ONE_DAY_POLL_INTERVAL_MS = 60000;
const ONE_DAY_HEARTBEAT_MS = 20000;
const ONE_DAY_DATA_ROOT = path.join(__dirname, "../app/json/one-day-price");

const oneDayRooms = new Map();
const oneDayClients = new Set();
let oneDayHeartbeatTimer = null;

const oneDayPriceCache = new Map();

function ensureOneDayHeartbeat() {
  if (!oneDayHeartbeatTimer) {
    oneDayHeartbeatTimer = setInterval(() => {
      for (const client of oneDayClients) {
        if (isSocketOpen(client.socket)) {
          try {
            client.socket.ping();
          } catch (err) {
            console.error("Failed to send one-day price heartbeat ping:", err);
          }
        }
      }
    }, ONE_DAY_HEARTBEAT_MS);
  }
}

function stopOneDayHeartbeatIfIdle() {
  if (oneDayHeartbeatTimer && oneDayClients.size === 0) {
    clearInterval(oneDayHeartbeatTimer);
    oneDayHeartbeatTimer = null;
  }
}

function ensureOneDayRoom(ticker) {
  const upper = ticker.toUpperCase();
  let room = oneDayRooms.get(upper);
  if (!room) {
    room = {
      ticker: upper,
      clients: new Set(),
      timer: null,
      lastPayload: null,
      lastSignature: null,
      pollInProgress: false,
    };
    oneDayRooms.set(upper, room);
  }
  if (!room.timer) {
    room.timer = setInterval(() => {
      pollOneDayRoom(room).catch((err) => {
        console.error(`Error polling one-day price data for ${room.ticker}:`, err);
      });
    }, ONE_DAY_POLL_INTERVAL_MS);
  }
  return room;
}

function stopOneDayRoomIfEmpty(room) {
  if (room.clients.size === 0) {
    if (room.timer) {
      clearInterval(room.timer);
      room.timer = null;
    }
    oneDayRooms.delete(room.ticker);
  }
}

async function pollOneDayRoom(room) {
  if (room.pollInProgress) {
    return;
  }
  if (room.clients.size === 0) {
    return;
  }
  if (!checkMarketHours()) {
    return;
  }

  room.pollInProgress = true;

  try {
    const result = await fetchOneDayPriceFromApi(room.ticker);
    if (!result || !Array.isArray(result.data) || result.data.length === 0) {
      return;
    }

    if (room.lastSignature === result.signature) {
      return;
    }

    room.lastPayload = result.data;
    room.lastSignature = result.signature;

    oneDayPriceCache.set(room.ticker, {
      data: result.data,
      timestamp: Date.now(),
      signature: result.signature,
    });

    await persistOneDayPrice(room.ticker, result.data);
    broadcastOneDayRoom(room, result.data, result.signature);
  } catch (err) {
    console.error(`Error fetching one-day price data for ${room.ticker}:`, err);
  } finally {
    room.pollInProgress = false;
  }
}

function broadcastOneDayRoom(room, data, signature) {
  const message = JSON.stringify(data);
  for (const client of room.clients) {
    if (!isSocketOpen(client.socket)) {
      continue;
    }
    if (client.lastSignature === signature) {
      continue;
    }
    try {
      client.socket.send(message);
      client.lastSignature = signature;
    } catch (err) {
      console.error("Failed to send one-day price update:", err);
    }
  }
}

async function sendOneDaySnapshot(client) {
  const room = client.room;
  if (!room) {
    return;
  }

  if (room.lastPayload && room.lastSignature) {
    sendJson(client.socket, room.lastPayload);
    client.lastSignature = room.lastSignature;
    return;
  }

  const cached = await loadCachedOneDayPrice(room.ticker);
  if (cached && cached.data) {
    room.lastPayload = cached.data;
    room.lastSignature = cached.signature;
    sendJson(client.socket, cached.data);
    client.lastSignature = cached.signature;
  }
}

async function loadCachedOneDayPrice(ticker) {
  const upper = ticker.toUpperCase();
  const cacheEntry = oneDayPriceCache.get(upper);
  if (cacheEntry) {
    return cacheEntry;
  }

  const filePath = path.join(ONE_DAY_DATA_ROOT, `${upper}.json`);
  try {
    const fileData = await fsp.readFile(filePath, "utf8");
    if (!fileData) {
      return null;
    }
    const payload = JSON.parse(fileData);
    const signature = buildOneDaySignature(payload);
    const entry = { data: payload, timestamp: Date.now(), signature };
    oneDayPriceCache.set(upper, entry);
    return entry;
  } catch (err) {
    if (err.code !== "ENOENT") {
      console.error(`Error loading cached one-day price for ${ticker}:`, err);
    }
    return null;
  }
}

async function persistOneDayPrice(ticker, data) {
  try {
    await fsp.mkdir(ONE_DAY_DATA_ROOT, { recursive: true });
    const filePath = path.join(ONE_DAY_DATA_ROOT, `${ticker}.json`);
    await fsp.writeFile(filePath, JSON.stringify(data, null, 2), "utf8");
  } catch (err) {
    console.error(`Error writing one-day price file for ${ticker}:`, err);
  }
}

async function joinOneDayRoom(client, ticker) {
  if (client.room && client.room.ticker === ticker) {
    await sendOneDaySnapshot(client);
    return;
  }

  removeClientFromOneDayRoom(client);
  const room = ensureOneDayRoom(ticker);
  room.clients.add(client);
  client.room = room;
  client.lastSignature = null;

  await sendOneDaySnapshot(client);
}

function removeClientFromOneDayRoom(client) {
  if (!client.room) {
    return;
  }
  const room = client.room;
  room.clients.delete(client);
  client.room = null;
  client.lastSignature = null;
  stopOneDayRoomIfEmpty(room);
}

function handleOneDayPrice(connection) {
  const client = {
    socket: connection.socket,
    room: null,
    lastSignature: null,
  };

  oneDayClients.add(client);
  ensureOneDayHeartbeat();

  const cleanup = () => {
    removeClientFromOneDayRoom(client);
    oneDayClients.delete(client);
    stopOneDayHeartbeatIfIdle();
  };

  connection.socket.on("message", (message) => {
    try {
      const payload = JSON.parse(message.toString("utf-8"));
      const tickerRaw = payload?.ticker;
      if (!tickerRaw) {
        return;
      }
      const ticker = String(tickerRaw).trim().toUpperCase();
      if (!ticker) {
        return;
      }
      joinOneDayRoom(client, ticker).catch((err) => {
        console.error(`Failed to join one-day price room for ${ticker}:`, err);
      });
    } catch (err) {
      console.error("Failed to parse one-day price message from client:", err);
    }
  });

  connection.socket.on("close", cleanup);
  connection.socket.on("error", (err) => {
    console.error("One-day price WebSocket error:", err);
    cleanup();
  });
}

function checkMarketHours() {
  const holidays = [
    "2025-01-01",
    "2025-01-09",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
  ];

  const currentTime = new Date().toLocaleString("en-US", {
    timeZone: "America/New_York",
  });
  const etDate = new Date(currentTime);

  const currentDateStr = etDate.toISOString().split("T")[0];
  const currentHour = etDate.getHours();
  const currentMinute = etDate.getMinutes();
  const currentDay = etDate.getDay();

  const isWeekend = currentDay === 0 || currentDay === 6;
  const isHoliday = holidays.includes(currentDateStr);

  if (isWeekend || isHoliday) {
    return false;
  }

  if (currentHour === 16 && currentMinute === 10) {
    return true;
  }

  return currentHour >= 9 && currentHour < 16;
}

function buildOneDaySignature(data) {
  if (!Array.isArray(data) || data.length === 0) {
    return "empty";
  }
  const last = data[data.length - 1];
  return `${data.length}-${last?.close}-${last?.time}`;
}

async function fetchOneDayPriceFromApi(ticker) {
  const apiKey = process.env.FMP_API_KEY;
  if (!apiKey) {
    console.error("FMP_API_KEY not found in environment variables");
    return null;
  }

  const today = new Date();
  const dateStr = today.toISOString().split("T")[0];
  const url = `https://financialmodelingprep.com/stable/historical-chart/1min?symbol=${ticker}&from=${dateStr}&to=${dateStr}&apikey=${apiKey}`;

  const response = await fetch(url);
  if (!response.ok) {
    console.error(`API request failed for ${ticker}: ${response.status}`);
    return null;
  }

  const rawData = await response.json();
  if (!Array.isArray(rawData) || rawData.length === 0) {
    return null;
  }

  const processedData = rawData
    .slice()
    .reverse()
    .filter((item) => item?.date && item.date.startsWith(dateStr))
    .map((item) => {
      const open = roundToTwo(item.open);
      const low = roundToTwo(item.low);
      const high = roundToTwo(item.high);
      const close = roundToTwo(item.close);
      if (
        open === null ||
        low === null ||
        high === null ||
        close === null
      ) {
        return null;
      }
      return {
        time: item.date,
        open,
        low,
        high,
        close,
      };
    })
    .filter(Boolean);

  if (processedData.length === 0) {
    return null;
  }

  const signature = buildOneDaySignature(processedData);
  return { data: processedData, signature };
}

const PRICE_DATA_INTERVAL_MS = 1000;
const PRICE_DATA_ROOT = path.join(
  __dirname,
  "../app/json/websocket/companies",
);

const priceDataConnections = new Set();
const priceDataRooms = new Map();
let priceDataTimer = null;

function ensurePriceDataTimer() {
  if (!priceDataTimer && priceDataRooms.size > 0) {
    priceDataTimer = setInterval(() => {
      runPriceDataPipeline().catch((err) => {
        console.error("Error running price data pipeline:", err);
      });
    }, PRICE_DATA_INTERVAL_MS);
  }
}

function stopPriceDataTimerIfIdle() {
  if (priceDataTimer && priceDataRooms.size === 0) {
    clearInterval(priceDataTimer);
    priceDataTimer = null;
  }
}

async function runPriceDataPipeline() {
  if (priceDataRooms.size === 0) {
    stopPriceDataTimerIfIdle();
    return;
  }

  const symbols = Array.from(priceDataRooms.keys());
  if (symbols.length === 0) {
    stopPriceDataTimerIfIdle();
    return;
  }

  const results = await Promise.all(
    symbols.map((symbol) => loadPriceDataForSymbol(symbol)),
  );

  for (let index = 0; index < symbols.length; index += 1) {
    const symbol = symbols[index];
    const room = priceDataRooms.get(symbol);
    if (!room) {
      continue;
    }

    room.justChanged = false;

    const result = results[index];
    if (!result) {
      if (room.lastPayload || room.lastSignature) {
        room.lastPayload = null;
        room.lastSignature = null;
        room.justChanged = true;
      }
      continue;
    }

    if (room.lastSignature !== result.signature) {
      room.lastSignature = result.signature;
      room.lastPayload = result.payload;
      room.justChanged = true;
    }
  }

  for (const client of priceDataConnections) {
    if (!isSocketOpen(client.socket) || client.symbols.size === 0) {
      continue;
    }

    const updates = [];
    for (const symbol of client.symbols) {
      const room = priceDataRooms.get(symbol);
      if (!room || !room.lastPayload || !room.lastSignature) {
        continue;
      }

      const previousSignature = client.lastSent.get(symbol);
      if (room.justChanged || previousSignature !== room.lastSignature) {
        updates.push(room.lastPayload);
        client.lastSent.set(symbol, room.lastSignature);
      }
    }

    if (updates.length > 0) {
      sendJson(client.socket, updates);
    }
  }
}

async function loadPriceDataForSymbol(symbol) {
  const upper = symbol.toUpperCase();
  const filePath = path.join(PRICE_DATA_ROOT, `${upper}.json`);

  try {
    const fileData = await fsp.readFile(filePath, "utf8");
    if (!fileData) {
      return null;
    }

    let jsonData;
    try {
      jsonData = JSON.parse(fileData);
    } catch (err) {
      console.error(`Invalid JSON format for ticker ${upper}:`, err);
      return null;
    }

    const { ap, bp, lp, t, type, ls } = jsonData;
    if (
      ap == null ||
      bp == null ||
      lp == null ||
      t == null ||
      (type !== "Q" && type !== "T")
    ) {
      return null;
    }

    const apNum = Number(ap);
    const bpNum = Number(bp);
    const lpNum = Number(lp);
    const timestamp = Number(t);

    if (
      !Number.isFinite(apNum) ||
      !Number.isFinite(bpNum) ||
      !Number.isFinite(lpNum) ||
      !Number.isFinite(timestamp)
    ) {
      return null;
    }

    const avgPrice = (apNum + bpNum + lpNum) / 3;
    const deviation = Math.abs(avgPrice - bpNum) / Math.abs(bpNum || 1);
    const finalPrice = deviation > 0.01 ? bpNum : avgPrice;

    if (Math.abs(finalPrice - avgPrice) > 1e-6) {
      return null;
    }

    const roundedFinalPrice = Number(finalPrice.toFixed(4));
    const payload = {
      symbol: upper,
      ap: apNum,
      bp: bpNum,
      lp: lpNum,
      ls,
      avgPrice: roundedFinalPrice,
      type,
      time: formatTimestampNewYork(timestamp),
    };

    const signature = `${upper}-${roundedFinalPrice}-${timestamp}`;
    return { symbol: upper, payload, signature };
  } catch (err) {
    if (err.code !== "ENOENT") {
      console.error(`Error processing data for ticker ${symbol}:`, err);
    }
    return null;
  }
}

async function sendPriceDataSnapshot(client) {
  if (!isSocketOpen(client.socket) || client.symbols.size === 0) {
    return;
  }

  const payload = [];

  for (const symbol of client.symbols) {
    const room = priceDataRooms.get(symbol);
    if (!room) {
      continue;
    }

    if (!room.lastPayload || !room.lastSignature) {
      const result = await loadPriceDataForSymbol(symbol);
      if (result) {
        room.lastPayload = result.payload;
        room.lastSignature = result.signature;
      }
    }

    if (room.lastPayload && room.lastSignature) {
      payload.push(room.lastPayload);
      client.lastSent.set(symbol, room.lastSignature);
    }
  }

  if (payload.length > 0) {
    sendJson(client.socket, payload);
  }
}

function removePriceDataConnection(client) {
  priceDataConnections.delete(client);

  for (const symbol of client.symbols) {
    const room = priceDataRooms.get(symbol);
    if (!room) {
      continue;
    }
    room.clients.delete(client);
    if (room.clients.size === 0) {
      priceDataRooms.delete(symbol);
    }
  }

  client.symbols.clear();
  client.lastSent.clear();

  stopPriceDataTimerIfIdle();
}

function updateClientSymbols(client, symbols) {
  const normalized = new Set(
    Array.isArray(symbols)
      ? symbols
          .map((symbol) => String(symbol || "").trim().toUpperCase())
          .filter(Boolean)
      : [],
  );

  const previousSymbols = new Set(client.symbols);

  for (const symbol of previousSymbols) {
    if (!normalized.has(symbol)) {
      const room = priceDataRooms.get(symbol);
      if (room) {
        room.clients.delete(client);
        if (room.clients.size === 0) {
          priceDataRooms.delete(symbol);
        }
      }
      client.lastSent.delete(symbol);
    }
  }

  for (const symbol of normalized) {
    if (!previousSymbols.has(symbol)) {
      let room = priceDataRooms.get(symbol);
      if (!room) {
        room = {
          symbol,
          clients: new Set(),
          lastSignature: null,
          lastPayload: null,
          justChanged: false,
        };
        priceDataRooms.set(symbol, room);
      }
      room.clients.add(client);
    }
  }

  client.symbols = normalized;

  if (priceDataRooms.size === 0) {
    stopPriceDataTimerIfIdle();
  } else {
    ensurePriceDataTimer();
  }
}

function handlePriceData(connection) {
  const client = {
    socket: connection.socket,
    symbols: new Set(),
    lastSent: new Map(),
  };

  priceDataConnections.add(client);

  const cleanup = () => {
    removePriceDataConnection(client);
  };

  connection.socket.on("message", (message) => {
    try {
      const payload = JSON.parse(message.toString("utf-8"));
      if (!Array.isArray(payload)) {
        return;
      }
      updateClientSymbols(client, payload);
      sendPriceDataSnapshot(client).catch((err) => {
        console.error("Failed to send price data snapshot:", err);
      });
    } catch (err) {
      console.error("Failed to parse tickers from client message:", err);
    }
  });

  connection.socket.on("close", cleanup);
  connection.socket.on("error", (err) => {
    console.error("Price data WebSocket error:", err);
    cleanup();
  });
}

fastify.register(async function (instance) {
  instance.get("/options-flow", { websocket: true }, handleOptionsFlow);
  instance.get("/pre-post-quote", { websocket: true }, handlePrePostQuote);
  instance.get("/one-day-price", { websocket: true }, handleOneDayPrice);
  instance.get("/price-data", { websocket: true }, handlePriceData);
  instance.get("/market-flow", { websocket: true }, handleMarketFlow);
});

let restartInProgress = false;

async function startServer() {
  if (serverRunning) {
    console.log("Server is already running.");
    return;
  }
  try {
    await fastify.listen({ port: 2000, host: "0.0.0.0" });
    serverRunning = true;
    console.log("Server started successfully on port 2000!");
  } catch (err) {
    console.error("Error starting server:", err);
    process.exit(1);
  }
}

async function stopServer() {
  if (!serverRunning) {
    console.log("Server is not running.");
    return;
  }
  try {
    await fastify.close();
    serverRunning = false;
    cleanupAllSchedulers();
    console.log("Server closed successfully!");
  } catch (err) {
    console.error("Error closing server:", err);
    throw err;
  }
}

async function restartServer() {
  if (restartInProgress) {
    return;
  }
  restartInProgress = true;
  try {
    await stopServer();
  } catch (err) {
    console.error("Failed to stop server during restart:", err);
  }
  try {
    await startServer();
  } finally {
    restartInProgress = false;
  }
}

function cleanupAllSchedulers() {
  if (optionsFlowTimer) {
    clearInterval(optionsFlowTimer);
    optionsFlowTimer = null;
  }
  optionsFlowSubscribers.clear();
  optionsFlowCache = [];
  optionsFlowRawSignature = null;

  for (const room of prePostRooms.values()) {
    if (room.timer) {
      clearInterval(room.timer);
      room.timer = null;
    }
    room.clients.clear();
  }
  prePostRooms.clear();

  if (oneDayHeartbeatTimer) {
    clearInterval(oneDayHeartbeatTimer);
    oneDayHeartbeatTimer = null;
  }
  for (const room of oneDayRooms.values()) {
    if (room.timer) {
      clearInterval(room.timer);
      room.timer = null;
    }
    room.clients.clear();
  }
  oneDayRooms.clear();
  oneDayClients.clear();
  oneDayPriceCache.clear();

  if (priceDataTimer) {
    clearInterval(priceDataTimer);
    priceDataTimer = null;
  }
  for (const room of priceDataRooms.values()) {
    room.clients.clear();
  }
  priceDataRooms.clear();
  priceDataConnections.clear();
}

process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:", err);
  restartServer().catch((restartErr) => {
    console.error("Failed to restart server after uncaught exception:", restartErr);
    process.exit(1);
  });
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
  restartServer().catch((restartErr) => {
    console.error("Failed to restart server after unhandled rejection:", restartErr);
    process.exit(1);
  });
});

startServer();
