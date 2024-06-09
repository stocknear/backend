import aiohttp
import asyncio
import sqlite3
import json
import ujson
import pandas as pd
import os
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime

import warnings
import requests

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')
coingecko_api_key = os.getenv('COINGECKO_API_KEY')

# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")



if os.path.exists("backup_db/crypto.db"):
    os.remove('backup_db/crypto.db')


def get_jsonparsed_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


def gecko_crypto_id(symbol):
    symbol_map = {
        'BTCUSD': 'bitcoin',
        'DASHUSD': 'dash',
        'ETCUSD': 'ethereum-classic',
        'SHIBUSD': 'shiba',
        'USDCUSD': 'usd-coin',
        'LINKUSD': 'chainlink',
        'BNBUSD': 'binancecoin',
        'ETHUSD': 'ethereum',
        'LTCUSD': 'litecoin',
        'SOLUSD': 'solana',
        'DOGEUSD': 'dogecoin',
        'XRPUSD': 'ripple',
        'XMRUSD': 'monero',
        'USDTUSD': 'tether',
        'ADAUSD': 'cardano',
        'AVAXUSD': 'avalanche-2',
        'LUNAUSD': 'terra-luna-2',
        'BCHUSD': 'bitcoin-cash',
        'TRXUSD': 'tron',
        'DOTUSD': 'polkadot',
        'ALGOUSD': 'algorand'
    }
    return symbol_map.get(symbol, None)

def get_description(symbol):
    symbol_map = {
        'BTCUSD': 'A brief history Bitcoin was created in 2009 by Satoshi Nakamoto, a pseudonymous developer. Bitcoin is designed to be completely decentralized and not controlled by any single authority. With a total supply of 21 million, its scarcity and decentralized nature make it almost impossible to inflate or manipulate. For this reason, many consider bitcoin to be the ultimate store of value or ‘Digital Gold’. Bitcoin is fully open-source and operates on a proof-of-work blockchain, a shared public ledger and history of transactions organized into "blocks" that are "chained" together to prevent tampering. This technology creates a permanent record of each transaction. Users on the Bitcoin network verify transactions through a process known as mining, which is designed to confirm new transactions are consistent with older transactions that have been confirmed in the past, ensuring users can not spend a Bitcoin they don’t have or attempt to double-spend coins.',
        'DASHUSD': 'Dash was launched in 2014 as a fork of Litecoin (which itself is a fork of Bitcoin). The founder, Evan Duffield, wanted to increase privacy and security in crypto transactions. The project was originally launched under the name "Darkcoin" before it got rebranded to Dash (Digital Cash) in 2015. Although the cryptocurrency still has robust encryption, the primary objective of the project has gone through readjustment. Its current aim is to provide an affordable and convenient means to make day-to-day payments for a wide range of products and services- a practical alternative to bank cards and hard cash. Dash distinguishes itself through its unique mining algorithms and its system for handling transactions. Dash uses the X11 algorithm, a modification of the proof-of-stake algorithm. It also uses CoinJoin mixing to scramble transactions and make privacy possible on its blockchain. Additionally, Dash is run by a subset of its users, which are called "masternodes." Masternodes simplify the verification and validation of transactions- reducing the number of nodes needed to successfully approve a transaction.',
        'ETCUSD': 'Ethereum Classic came into existence on July 20, 2016, as a continuation of the original Ethereum blockchain following a compromise in the original protocol leading to a fork of the protocol. Ethereum Classic is dedicated to enabling decentralized, immutable, and unstoppable applications. Like the original Ethereum network, the blockchain relies on "proof of work" mining, meaning that users run hardware and software to validate transactions on the network and keep it secure- earning ETC in return. However, Ethereum Classic differs from Ethereum in that the platform does not plan to move away from Proof-of-Work, while Ethereum is trying to make the transition to Proof-of-Stake It should also be noted that the Ethereum Classic ecosystem is not as active as the Ethereum network. These relatively low rates of use have caused problems for the networks security since blockchains rely on having a distributed group of users running the network; when there are not enough people actively doing so, it leaves the blockchain vulnerable. However, Ethereum Classic has been actively making updates to address this issue in its network.',
        'LINKUSD': 'Chainlink was created by Sergey Nazarov and Steve Ellis, who authored a 2017 white paper with Ari Juels. Launched in 2019, Chainlink is described as a decentralized "oracle" network which aims to bring real-world data onto the blockchain. Oracles are entities that connect blockchains to external systems. Though traditional oracles are centralized, Chainlink decentralizes the process of moving data on and off blockchains through the use of "hybrid smart contracts." These hybrid smart contracts create access to off-chain resources, letting them react to real-world events and execute agreements that would otherwise need external proof of performance. As a result, Chainlink has been used to distribute non-fungible tokens (NFTs), gamify personal savings, and facilitate recalibrations of cryptocurrency token supplies, among other applications.',
        'USDCUSD': 'USD Coin (USDC) was launched in September of 2018 by Center — a joint venture between Coinbase and Circle. USDC first launched on the Ethereum blockchain as an ERC-20 token, but has since expanded to other blockchains including Solana, Stellar, and Algorand, and can be purchased on both centralized and decentralized exchanges (DEXs). As a stablecoin, it provides all the benefits of cryptocurrencies––faster, cheaper, permissionless transactions––without the price volatility.',
        'SHIBUSD': 'Launched in August 2020 by a founder called Ryoshi, Shiba Inu (SHIB) was created as an Ethereum-based meme coin inspired by Dogecoin. According to the project`s “woofpaper” (whitepaper), Shiba Inu was developed as the answer to a simple question: What would happen if a cryptocurrency project was 100% run by its community? Its founder Ryoshi attributes its origins to an "experiment in decentralized spontaneous community building. Since its founding, it has evolved into a decentralized ecosystem supporting projects such as an NFT art incubator and a decentralized exchange called Shibaswap.',
        'BNBUSD': 'Binance was founded in 2017 by Changpeng Zhao, a developer who had previously created a high-frequency trading software called Fusion Systems. Binance was initially based in China but later moved its headquarters following the Chinese government`s increasing regulation of cryptocurrency. Binance offers crypto-to-crypto trading in more than 500 cryptocurrencies and virtual tokens, with a strong focus on altcoin trading. Additionally, Binance has among the lowest transaction fees for cryptocurrency exchanges thanks to its commission structure. Fees generally start low, and then only move lower. Binance uses a volume-based pricing model and even gives you further discounts if you use its proprietary cryptocurrency to buy and sell.',
        'ETHUSD': 'The original Ethereum concept was introduced in 2013 by Vitalik Buterin with the release of the Ethereum whitepaper and in 2015 the Ethereum platform was launched by Buterin and Joseph Lubin along with several other co-founders. Ethereum is described as “the world’s programmable blockchain,” positioning itself as an electronic, programmable network that anyone can build on to launch cryptocurrencies and decentralized applications. Unlike Bitcoin which has a maximum circulation of 21 million coins, the amount of ETH that can be created is unlimited, although the time that it takes to process a block of ETH limits how much ether can be minted each year. Another difference between Ethereum and Bitcoin is how the networks treat transaction processing fees. These fees are known as “gas” on the Ethereum network and are paid by the participants in Ethereum transactions. The fees associated with Bitcoin transactions, however, are absorbed by the broader Bitcoin network. Additionally, although both Bitcoin and Ethereum currently use Proof-of-Work consensus mechanisms, Ethereum is in the process of gradually transitioning to a different consensus algorithm known as Proof-of-Stake, which uses significantly less energy.',
        'LTCUSD': 'Litecoin was launched in 2011 by Charlie Lee, a former Google employee. It aims to be a "lite" version of Bitcoin in that it features many of the same properties as Bitcoin–albeit lighter weight. It is commonly often referred to as digital silver to Bitcoins digital gold and is often used as a pseudo testnet for Bitcoin, adopting new protocol changes before they are deployed on Bitcoin. Like Bitcoin, Litecoin uses a form of proof-of-work mining to enable anyone who dedicates their computing resources to add new blocks to its blockchain and earn the new Litecoin it creates. Where Litecoin differs is in its mining algorithm called Scrypt PoW. Scrypt allows the platform to finalize transactions faster. On Litecoin, new blocks are added to the blockchain roughly every 2.5 minutes (as opposed to 10 minutes on Bitcoin).',
        'SOLUSD': 'Solana was created in 2017 by Anatoly Yakovenko and Raj Gokal. Yakovenko, who is also the CEO of Solana Labs, came from a background in system design and wanted to apply this knowledge and create a brand new blockchain that could scale to global adoption. Solana boasts a theoretical peak capacity of 65,000 transactions per second and has become one of the most highly used blockchains due to its speed and low transaction costs. Solana runs on a hybrid protocol of proof-of-stake (PoS) and a concept Solana calls proof-of-history (PoH). Solana is also said to be an “Ethereum competitor,” due to its distinct advantage over Ethereum in terms of transaction processing speed and transaction costs. Solana can process as many as 50,000 transactions per second (TPS), and its average cost per transaction is $0.00025. In contrast, Ethereum can only handle less than 15 TPS, while transaction fees reached a record of $70 in 2021.',
        'DOGEUSD': 'Founded in 2013 by software engineers Billy Markus and Jackson Palmer, Dogecoin was created as a marketing experiment based on the popular "Doge" Internet meme and as a lighthearted alternative to traditional cryptocurrencies. Despite its origins as a “joke,” Dogecoin’s scrypt technology (a hashing function that uses SHA-256 but includes much higher memory requirements for proof-of-work) and an unlimited supply of coins set it apart as a faster, more adaptable, and consumer-friendly version of Bitcoin. Like other cryptocurrencies, Dogecoin is mined by the decentralized network of computers that runs the currency. But unlike other coins, Dogecoin does not have a cap on the total number of coins that can be mined- making it an inflationary rather than deflationary coin. In 2014 due to network security issues, Dogecoin agreed to merge mine its network with Litecoin (LTC).',
        'XRPUSD': 'RippleNet was founded in 2012 by Chris Larsen and Jed McCaleb and is based on the work of Ryan Fugger, who created the XRP Ledger- an open-source cryptographic ledger powered by a peer-to-peer network of nodes. XRP’s main aim is to increase the speed and reduce the cost of transferring money between financial institutions. XRP does this through an open-source and peer-to-peer decentralized platform that allows for a seamless transfer of money in any form. XRP is a global network and counts major banks and financial services among its customers. Ripple uses a medium, known as a gateway, as the link in the trust chain between two parties wanting to make a transaction. Usually, in the form of banks, the gateway acts as a trusted intermediary to help two parties complete a transaction by providing a channel to transfer funds in fiat and cryptocurrencies. It should also be noted that XRP runs a federated consensus algorithm which differs from both Proof-of-Work and Proof-of-Stake mechanisms. Essentially, the mechanism allows participating nodes to validate transactions by conducting a poll, enabling almost instant confirmations without a central authority.',
        'XMRUSD': 'Monero, originally called Bitmonero, was launched in 2014 after a hard fork from Bytecoin. Monero (XMR) is a cryptocurrency focused on privacy. It aims to allow payments to be made quickly and inexpensively without fear of censorship. Monero is unique in that it’s designed to keep wallets and transactions completely anonymous, including network members, developers, and miners. Monero alleviates privacy concerns using the concepts of ring signatures and stealth addresses. Ring signatures enable a sender to conceal their identity from other participants in a group. To generate a ring signature, the Monero platform uses a combination of a sender’s account keys and combines it with public keys on the blockchain, making it unique as well as private. It hides the senders identity, as it is computationally impossible to ascertain which of the group members keys was used to produce the complex signature.',
        'USDTUSD': 'Originally known as “Realcoin,” Tether was founded in July 2014 by Brock Pierce, Craig Sellars, and Reeve Collins. Tether aims to solve two major issues with existing cryptocurrencies: high volatility and convertibility between fiat currencies and cryptocurrencies. To address these perceived issues Tether created a cryptocurrency that is fully backed 1:1 by deposits of fiat currencies like the US dollar, the euro, or the yen. This makes Tether a fiat-based stablecoin, which differs from other stablecoins such as crypto-collateralized stablecoins, which use cryptocurrency reserves as collateral. Tether relies on a Proof-of-Reserve to ensure that reserve assets match circulating USTD tokens. Doing this requires a third party to audit Tether’s bank accounts on a regular basis to show that the reserves are held in an amount equal to the outstanding tokens. Tether uses an IOU model where each USDT represents a claim for $1.00 held in Tether’s reserves.',
        'ADAUSD': 'Cardano is a blockchain founded on peer-reviewed research by Charles Hoskinson, a co-founder of the Ethereum project. He began developing Cardano in 2015, launching the platform and the ADA token in 2017. Positioned as an alternative to Ethereum, Cardano aims to offer greater security, scalability, and energy efficiency than its peers. Currently, Cardano has released three products: Atala PRISM, Atala SCAN, and Atala Trace. The first product is marketed as an identity management tool that can be used to provide access to services, while the other two products are being used to trace a product’s journey through a supply chain. Additionally, Cardano utilizes Ouroboros, an algorithm that uses proof-of-stake (PoS) protocol to mine blocks. The protocol is designed to reduce energy expenditure during the block production process to a minimum by eliminating the need for massive computing resources that are more central to the functioning of the proof-of-work (PoW) protocol. In Cardanos PoS system, staking determines a nodes capability to create blocks, and a nodes stake is equal to the amount of ADA held by it over the long term.',
        'AVAXUSD': 'Launched in 2020 by the Ava Labs team, Avalanche quickly ascended the cryptocurrency rankings while aiming to be the fastest, lowest cost, and most environmentally-friendly blockchain. Although Avalanche’s platform is complex, there are three primary aspects of its design that distinguish it from other blockchain projects. First, it uses a novel consensus mechanism that builds off of PoS. When a transaction is received by a validator node that node then samples a random set of other validators (which then randomly samples another set of validators) and checks for agreement until consensus is reached. Second, Avalanche users can launch specialized chains called sub-nets that operate using their own sets of rules- comparable to Polkadot’s parachains and Ethereum 2.0’s shards. Lastly, Avalanche is built using three different blockchains called the X-Chain, C-Chain, and P-Chain. Digital assets can be moved across each of these chains to accomplish different functions within the ecosystem.',
        'BCHUSD': 'Bitcoin Cash came about in 2017 and was created to address concerns over Bitcoins scalability while staying as close to its original vision as a form of digital cash. It’s a hard fork of the Bitcoin blockchain, meaning the network “split” in two at a certain block as decided on by various miners and developers within the Bitcoin network. Bitcoin Cash uses an increased block size with an adjustable level of difficulty to ensure fast transactions as its user base scales. At a technical level, Bitcoin Cash works exactly the same as Bitcoin. Both platforms have a hard cap of 21 million assets, use nodes to validate transactions, and use a PoW consensus algorithm. However, BCH operates faster and has lower transaction fees than its predecessor, thanks to the aforementioned larger block size. Bitcoin Cash can support 25,000 transactions per block compared with Bitcoin’s 1,000 to 1,500 per block. Additionally, as of March 2022, the maximum block size for BCH was increased fourfold to 32 MB.',
        'TRXUSD': 'Founded in 2017 by a Singapore non-profit organization, the Tron Foundation, Tron aims to host a global entertainment system and to be the infrastructure of the decentralized web. It powers an ecosystem of decentralized applications (DApps) by offering high throughput, high scalability, and high availability. The Tron network relies on a Delegated-Proof-of-Stake (DPoS) consensus mechanism to secure the blockchain. Similar to a proof-of-stake consensus mechanism, DPoS allows users to earn passive income whenever they stake their holdings in a network wallet. However, unlike a PoS system, only elected nodes can approve blocks of transactions on Tron. In the Tron ecosystem, the community elects 27 Super Representatives every six hours for this vital task. These representatives are chosen by the community of users staking their TRX. In addition to super representatives, users can operate as witness nodes, full nodes and Solidity nodes. Witness nodes propose blocks and vote on protocol decisions, full nodes broadcast transactions and blocks, and Solidity nodes sync blocks from full nodes and provide APIs. It should also be noted that there are no transaction fees for TRX, and the network is able to support 2000 of these feeless transactions every second.',
        'DOTUSD': 'Polkadot was launched on May 26, 2020, by Ethereum co-creator Gavin Wood, with the nonprofit Web3 Foundation serving as the primary research organization that maintains Polkadot’s open-source code. The main goal of the Polkadot platforms is to connect blockchains and enable cross-blockchain transfers of data and assets. Polkadot does this through an innovative "parallel chain," or para chain, system to enable interoperability between blockchains. The Polkadot platform allows for the creation of three types of blockchains. At the centre is the Relay chain- the main Polkadot blockchain- which is where transactions are finalized. On top of that, there are Parachains -custom blockchains- that use the relay chain’s computing resources to confirm that transactions are accurate. Lastly, there are Bridges which allow the Polkadot network to interact with other blockchains.',
        'ALGOUSD': 'Algorand was founded by Silvio Micali, a professor of computer science at the Massachusetts Institute of Technology and the 2012 recipient of the Turing Award. Launched in 2019, Algorand is a protocol that is designed to solve the blockchain trilemma of achieving speed, security, and decentralization simultaneously. In doing so, it aims to power the future of finance (FutureFi) by accelerating the convergence of decentralized and traditional finance (DeFi and TradFi). Additionally, the Algorand blockchain uses a modified version of the proof-of-stake consensus mechanism called pure-proof-of-stake (PPoS). The PPoS consensus mechanism uses randomness to increase the security of the blockchain network. While the traditional proof-of-stake method enables cryptocurrency holders to opt into validating transactions, the pure-proof-of-stake method recruits validators from the entire pool of ALGO holders.',
    }
    return symbol_map.get(symbol, 'No description available')

def get_website(symbol):
    symbol_map = {
        'BTCUSD': 'https://bitcoin.org',
        'DASHUSD': 'https://www.dash.org/',
        'ETCUSD': 'https://ethereumclassic.org/',
        'LINKUSD': 'https://chain.link/',
        'USDCUSD': 'https://www.centre.io/usdc',
        'SHIBUSD': 'https://shibatoken.com/',
        'BNBUSD': 'https://www.bnbchain.org',
        'ETHUSD': 'https://ethereum.org',
        'LTCUSD': 'https://litecoin.org/',
        'SOLUSD': 'https://solana.com/',
        'DOGEUSD': 'https://dogecoin.com/',
        'XRPUSD': 'https://xrpl.org/',
        'XMRUSD': 'https://www.getmonero.org/',
        'USDTUSD': 'https://tether.to',
        'ADAUSD': 'https://cardano.org/',
        'AVAXUSD': 'https://avax.network/',
        'LUNAUSD': '/',
        'BCHUSD': 'https://bch.info',
        'TRXUSD': 'https://tron.network/',
        'DOTUSD': 'https://polkadot.network/',
        'ALGOUSD': 'https://algorandtechnologies.com/'
    }
    return symbol_map.get(symbol, '/')

class CryptoDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode = wal")
        self.conn.commit()
        self._create_table()

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS cryptos (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            type TEXT
        )
        """)

    def get_column_type(self, value):
        column_type = ""

        if isinstance(value, str):
            column_type = "TEXT"
        elif isinstance(value, int):
            column_type = "INTEGER"
        elif isinstance(value, float):
            column_type = "REAL"
        else:
            # Handle other data types or customize based on your specific needs
            column_type = "TEXT"

        return column_type

    def remove_null(self, value):
        if isinstance(value, str) and value == None:
            value = 'n/a'
        elif isinstance(value, int) and value == None:
            value = 0
        elif isinstance(value, float) and value == None:
            value = 0
        else:
            # Handle other data types or customize based on your specific needs
            pass

        return value

    async def save_fundamental_data(self, session, symbol):
        try:
            crypto_id = gecko_crypto_id(symbol)

            urls = [
                f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/crypto_news?tickers={symbol}&limit=50&apikey={api_key}",
                f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={crypto_id}"
            ]

            fundamental_data = {}

   
            for url in urls:
                async with session.get(url) as response:
                    data = await response.text()
                    parsed_data = get_jsonparsed_data(data)

                    try:
                        if isinstance(parsed_data, list) and "quote" in url:
                            fundamental_data['quote'] = ujson.dumps(parsed_data)
                            data_dict = {
                                        'price': parsed_data[0]['price'],
                                        'changesPercentage': round(parsed_data[0]['changesPercentage'],2),
                                        'marketCap': parsed_data[0]['marketCap'],
                                        'previousClose': parsed_data[0]['previousClose'],
                                        }
                            fundamental_data.update(data_dict)

                        elif isinstance(parsed_data, list) and "crypto_news" in url:
                            fundamental_data['crypto_news'] = ujson.dumps(parsed_data)
                        elif "coingecko" in url:
                            headers = {
                                "accept": "application/json",
                                "x-cg-demo-api-key": coingecko_api_key
                            }
                            response = requests.get(url, headers=headers)

                            gecko_data = ujson.loads(response.text)[0]
                            gecko_data['description'] = get_description(symbol)
                            gecko_data['website'] = get_website(symbol)
                            fundamental_data['profile'] = ujson.dumps(gecko_data)

                            max_supply = gecko_data.get('max_supply')
                            if max_supply is None:
                                max_supply = "Uncapped"

                            data_dict = {
                                        'circulatingSupply': gecko_data['circulating_supply'],
                                        'maxSupply': max_supply,
                                        }
                            fundamental_data.update(data_dict)

                    except:
                        pass


            # Check if columns already exist in the table
            self.cursor.execute("PRAGMA table_info(cryptos)")
            columns = {column[1]: column[2] for column in self.cursor.fetchall()}

            # Update column definitions with keys from fundamental_data
            column_definitions = {
                key: (self.get_column_type(fundamental_data.get(key, None)), self.remove_null(fundamental_data.get(key, None)))
                for key in fundamental_data
            }

           
            for column, (column_type, value) in column_definitions.items():
                if column not in columns and column_type:
                    self.cursor.execute(f"ALTER TABLE cryptos ADD COLUMN {column} {column_type}")

                self.cursor.execute(f"UPDATE cryptos SET {column} = ? WHERE symbol = ?", (value, symbol))

            self.conn.commit()

        except Exception as e:
            print(f"Failed to fetch fundamental data for symbol {symbol}: {str(e)}")



    async def save_cryptos(self, cryptos):
        symbols = []
        names = []
        ticker_data = []

        for item in cryptos:
            symbol = item.get('symbol', '')
            name = item.get('name', '').replace('USDt','').replace('USD','')
            exchange = item.get('exchangeShortName', '')
            ticker_type = 'crypto'

            if name and '.' not in symbol and not re.search(r'\d', symbol):
                symbols.append(symbol)
                names.append(name)
                ticker_data.append((symbol, name, exchange, ticker_type))
        

        self.cursor.execute("BEGIN TRANSACTION")  # Begin a transaction

        for data in ticker_data:
            symbol, name, exchange, ticker_type = data

            self.cursor.execute("""
            INSERT OR IGNORE INTO cryptos (symbol, name, exchange, type)
            VALUES (?, ?, ?, ?)
            """, (symbol, name, exchange, ticker_type))
            self.cursor.execute("""
            UPDATE cryptos SET name = ?, exchange = ?, type= ?
            WHERE symbol = ?
            """, (name, exchange, ticker_type, symbol))

        self.cursor.execute("COMMIT")  # Commit the transaction
        self.conn.commit()

    

        # Save OHLC data for each ticker using aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = []
            i = 0
            for item in tqdm(ticker_data):
                symbol, name, exchange, ticker_type = item
                symbol = symbol.replace("-", "")
                tasks.append(self.save_ohlc_data(session, symbol))
                tasks.append(self.save_fundamental_data(session, symbol))

                i += 1
                if i % 150 == 0:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print('sleeping mode: ', i)
                    await asyncio.sleep(60)  # Pause for 60 seconds

            #tasks.append(self.save_ohlc_data(session, "%5EGSPC"))
            
            if tasks:
                await asyncio.gather(*tasks)


    def _create_ticker_table(self, symbol):
        #cleaned_symbol = re.sub(r'[^a-zA-Z0-9_]', '_', symbol)
        # Check if table exists
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'")
        table_exists = self.cursor.fetchone() is not None

        if not table_exists:
            query = f"""
            CREATE TABLE '{cleaned_symbol}' (
                date TEXT,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume INT,
                change_percent FLOAT,
            );
            """
            self.cursor.execute(query)

    async def save_ohlc_data(self, session, symbol):
        try:
            #self._create_ticker_table(symbol)  # Create table for the symbol

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=bar&from={start_date}&to={end_date}&apikey={api_key}"

            try:
                async with session.get(url) as response:
                    data = await response.text()

                ohlc_data = get_jsonparsed_data(data)
                if 'historical' in ohlc_data:
                    ohlc_values = [(item['date'], item['open'], item['high'], item['low'], item['close'], item['volume'], item['changePercent']) for item in ohlc_data['historical'][::-1]]

                    df = pd.DataFrame(ohlc_values, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'change_percent'])
                
                    # Perform bulk insert
                    df.to_sql(symbol, self.conn, if_exists='append', index=False)

            except Exception as e:
                print(f"Failed to fetch OHLC data for symbol {symbol}: {str(e)}")
        except Exception as e:
            print(f"Failed to create table for symbol {symbol}: {str(e)}")


url = f"https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies?apikey={api_key}"

async def fetch_tickers():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.text()
            return get_jsonparsed_data(data)


db = CryptoDatabase('backup_db/crypto.db')
loop = asyncio.get_event_loop()
all_tickers = [item for item in loop.run_until_complete(fetch_tickers()) if item['symbol'] in ['DASHUSD','ETCUSD','LINKUSD','USDCUSD','SHIBUSD','BNBUSD','BTCUSD', 'ETHUSD', 'LTCUSD', 'SOLUSD','DOGEUSD','XRPUSD','XMRUSD','USDTUSD','ADAUSD','AVAXUSD','BCHUSD','TRXUSD','DOTUSD','ALGOUSD']]

loop.run_until_complete(db.save_cryptos(all_tickers))
db.close_connection()